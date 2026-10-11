#include "collective.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>

namespace dist {

DistContext::DistContext(int rank_, int world_size_, const std::vector<RankInfo>& infos)
    : rank(rank_), world_size(world_size_), rank_infos(infos) {
    peers.resize(world_size);
}

bool DistContext::initialize() {
    const RankInfo& my_info = rank_infos[rank];
    // Rank 0 acts as coordinator: others connect to rank 0, then rank 0 redistributes connections
    // Simplified: each rank listens on its port; lower ranks connect to higher ranks
    std::vector<std::thread> threads;

    // Listen thread
    net::ServerSocket listener;
    bool listen_ok = listener.bind_and_listen("0.0.0.0", my_info.port);
    if (!listen_ok) {
        std::cerr << "Rank " << rank << " failed to listen on port " << my_info.port << "\n";
        return false;
    }

    std::atomic<int> accepted{0};
    std::thread listen_thread([&]() {
        int expected_accept = 0;
        for (int r = 0; r < world_size; ++r) {
            if (r > rank) expected_accept++;
        }
        while (accepted.load() < expected_accept) {
            net::Socket client = listener.accept_connection();
            if (!client.valid()) continue;
            // Receive rank id from connecting peer
            uint32_t peer_rank_be = 0;
            if (!client.recv_all(reinterpret_cast<uint8_t*>(&peer_rank_be), sizeof(peer_rank_be))) continue;
            int peer_rank = static_cast<int>(ntohl(peer_rank_be));
            if (peer_rank >= 0 && peer_rank < world_size) {
                std::lock_guard<std::mutex> lock(mtx);
                peers[peer_rank] = std::move(client);
                accepted.fetch_add(1);
            }
        }
    });

    // Connect to all lower ranks
    for (int r = 0; r < rank; ++r) {
        bool connected = false;
        for (int retry = 0; retry < 30 && !connected; ++retry) {
            net::Socket s = net::connect_to(rank_infos[r].host, rank_infos[r].port, 5000);
            if (s.valid()) {
                // Send my rank
                uint32_t my_rank_be = htonl(static_cast<uint32_t>(rank));
                if (s.send_all(reinterpret_cast<uint8_t*>(&my_rank_be), sizeof(my_rank_be))) {
                    std::lock_guard<std::mutex> lock(mtx);
                    peers[r] = std::move(s);
                    connected = true;
                }
            }
            if (!connected) std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (!connected) {
            std::cerr << "Rank " << rank << " failed to connect to rank " << r << "\n";
            listen_thread.join();
            return false;
        }
    }

    listen_thread.join();
    std::cout << "Rank " << rank << " connected to all " << world_size - 1 << " peers.\n";
    return true;
}

// ---------- Ring All-Reduce ----------
bool DistContext::all_reduce_sum_ring(Tensor& tensor) {
    if (world_size == 1) return true;
    size_t n = tensor.numel();
    if (n == 0) return true;

    int left = (rank - 1 + world_size) % world_size;
    int right = (rank + 1) % world_size;

    size_t chunk_size = (n + world_size - 1) / world_size;
    std::vector<float> recv_buf(chunk_size, 0.0f);

    // Phase 1: Scatter-Reduce
    std::vector<float> local_chunks(world_size * chunk_size, 0.0f);
    for (int i = 0; i < world_size; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        for (size_t j = start; j < end; ++j) {
            local_chunks[i * chunk_size + (j - start)] = tensor.data[j];
        }
    }

    for (int step = 0; step < world_size - 1; ++step) {
        int send_chunk_idx = (rank - step + world_size) % world_size;
        int recv_chunk_idx = (rank - step - 1 + world_size) % world_size;

        size_t send_start = send_chunk_idx * chunk_size;
        size_t send_count = std::min(chunk_size, n - send_start);

        // Send to right, receive from left simultaneously using threads
        std::thread send_thread([&]() {
            peers[right].send_floats(std::vector<float>(
                local_chunks.begin() + send_chunk_idx * chunk_size,
                local_chunks.begin() + send_chunk_idx * chunk_size + send_count));
        });

        std::vector<float> received;
        bool ok = peers[left].recv_floats(received, std::min(chunk_size, n - recv_chunk_idx * chunk_size));
        if (!ok) {
            send_thread.join();
            return false;
        }
        send_thread.join();

        // Accumulate
        size_t recv_offset = recv_chunk_idx * chunk_size;
        size_t recv_count = std::min(chunk_size, n - recv_offset);
        for (size_t i = 0; i < recv_count; ++i) {
            local_chunks[recv_chunk_idx * chunk_size + i] += received[i];
        }
    }

    // Phase 2: All-Gather
    for (int step = 0; step < world_size - 1; ++step) {
        int send_chunk_idx = (rank - step - 1 + world_size) % world_size;
        int recv_chunk_idx = (rank - step + world_size) % world_size;

        size_t send_count = std::min(chunk_size, n - send_chunk_idx * chunk_size);

        std::thread send_thread([&]() {
            peers[left].send_floats(std::vector<float>(
                local_chunks.begin() + send_chunk_idx * chunk_size,
                local_chunks.begin() + send_chunk_idx * chunk_size + send_count));
        });

        std::vector<float> received;
        bool ok = peers[right].recv_floats(received, std::min(chunk_size, n - recv_chunk_idx * chunk_size));
        if (!ok) {
            send_thread.join();
            return false;
        }
        send_thread.join();

        size_t recv_count = std::min(chunk_size, n - recv_chunk_idx * chunk_size);
        for (size_t i = 0; i < recv_count; ++i) {
            local_chunks[recv_chunk_idx * chunk_size + i] = received[i];
        }
    }

    // Write back
    for (int i = 0; i < world_size; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        for (size_t j = start; j < end; ++j) {
            tensor.data[j] = local_chunks[i * chunk_size + (j - start)];
        }
    }
    return true;
}

// Simple all-reduce: reduce to rank 0, then broadcast (less efficient but simpler)
bool DistContext::all_reduce_sum(Tensor& tensor) {
    if (world_size == 1) return true;
    size_t n = tensor.numel();
    if (n == 0) return true;

    // Reduce to rank 0
    if (rank != 0) {
        if (!peers[0].send_floats(tensor.data)) return false;
    } else {
        std::vector<float> buf(n);
        for (int r = 1; r < world_size; ++r) {
            if (!peers[r].recv_floats(buf, n)) return false;
            for (size_t i = 0; i < n; ++i) tensor.data[i] += buf[i];
        }
        // Average
        for (size_t i = 0; i < n; ++i) tensor.data[i] /= static_cast<float>(world_size);
    }

    // Broadcast from rank 0
    return broadcast(tensor, 0);
}

bool DistContext::broadcast(Tensor& tensor, int root) {
    if (world_size == 1) return true;
    size_t n = tensor.numel();
    if (rank == root) {
        for (int r = 0; r < world_size; ++r) {
            if (r == root) continue;
            if (!peers[r].send_floats(tensor.data)) return false;
        }
    } else {
        if (!peers[root].recv_floats(tensor.data, n)) return false;
    }
    return true;
}

bool DistContext::barrier() {
    uint8_t token = 1;
    if (rank == 0) {
        for (int r = 1; r < world_size; ++r) {
            uint8_t t = 0;
            if (!peers[r].recv_all(&t, 1)) return false;
        }
        for (int r = 1; r < world_size; ++r) {
            if (!peers[r].send_all(&token, 1)) return false;
        }
    } else {
        if (!peers[0].send_all(&token, 1)) return false;
        uint8_t t = 0;
        if (!peers[0].recv_all(&t, 1)) return false;
    }
    return true;
}

bool DistContext::average_gradients(std::vector<Tensor*>& grads) {
    for (auto* g : grads) {
        if (!all_reduce_sum_ring(*g)) return false;
        // all_reduce_sum_ring already divides by world_size in the reduce-to-root path,
        // but for ring allreduce we need to divide manually.
        float inv_ws = 1.0f / static_cast<float>(world_size);
        for (auto& v : g->data) v *= inv_ws;
    }
    return true;
}

bool DistContext::send_tensor(int dst_rank, const Tensor& tensor) {
    // Send shape + data
    std::vector<uint8_t> msg;
    uint32_t ndim = static_cast<uint32_t>(tensor.ndim());
    msg.resize(sizeof(uint32_t) + ndim * sizeof(uint32_t) + tensor.numel() * sizeof(float));
    size_t off = 0;
    std::memcpy(msg.data() + off, &ndim, sizeof(ndim)); off += sizeof(ndim);
    for (size_t i = 0; i < tensor.ndim(); ++i) {
        uint32_t s = static_cast<uint32_t>(tensor.shape[i]);
        std::memcpy(msg.data() + off, &s, sizeof(s)); off += sizeof(s);
    }
    std::memcpy(msg.data() + off, tensor.data.data(), tensor.numel() * sizeof(float));
    return peers[dst_rank].send_message(msg);
}

bool DistContext::recv_tensor(int src_rank, Tensor& tensor) {
    std::vector<uint8_t> msg;
    if (!peers[src_rank].recv_message(msg)) return false;
    size_t off = 0;
    uint32_t ndim = 0;
    std::memcpy(&ndim, msg.data() + off, sizeof(ndim)); off += sizeof(ndim);
    std::vector<size_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t s = 0;
        std::memcpy(&s, msg.data() + off, sizeof(s)); off += sizeof(s);
        shape[i] = static_cast<size_t>(s);
    }
    tensor = Tensor(shape, false);
    size_t num_bytes = tensor.numel() * sizeof(float);
    if (off + num_bytes != msg.size()) return false;
    std::memcpy(tensor.data.data(), msg.data() + off, num_bytes);
    return true;
}

std::vector<RankInfo> parse_rank_config(const std::string& config_path) {
    std::vector<RankInfo> infos;
    // Simple format: each line "rank host port"
    FILE* f = fopen(config_path.c_str(), "r");
    if (!f) return infos;
    int rank; char host[256]; int port;
    while (fscanf(f, "%d %255s %d", &rank, host, &port) == 3) {
        infos.push_back({rank, std::string(host), port});
    }
    fclose(f);
    // Sort by rank
    std::sort(infos.begin(), infos.end(), [](const RankInfo& a, const RankInfo& b) {
        return a.rank < b.rank;
    });
    return infos;
}

} // namespace dist
