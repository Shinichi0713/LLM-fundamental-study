#pragma once
#include "tensor.hpp"
#include "network.hpp"
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace dist {

struct RankInfo {
    int rank = -1;
    std::string host;
    int port = 0;
};

// Distributed context managing all peer connections
class DistContext {
public:
    int rank = -1;
    int world_size = 1;
    std::vector<RankInfo> rank_infos;
    std::vector<net::Socket> peers; // connections to all other ranks
    std::mutex mtx;

    DistContext(int rank_, int world_size_, const std::vector<RankInfo>& infos);
    ~DistContext() = default;

    // Initialize connections (blocking)
    bool initialize();

    // Collective operations (blocking, synchronous)
    bool all_reduce_sum(Tensor& tensor);
    bool all_reduce_sum_ring(Tensor& tensor);
    bool broadcast(Tensor& tensor, int root = 0);
    bool barrier();

    // Parameter sync: average gradients across all ranks
    bool average_gradients(std::vector<Tensor*>& grads);

    // Non-blocking send/recv for pipeline parallelism
    bool send_tensor(int dst_rank, const Tensor& tensor);
    bool recv_tensor(int src_rank, Tensor& tensor);

    bool is_root() const { return rank == 0; }
};

// Helper to parse rank config from env or file
std::vector<RankInfo> parse_rank_config(const std::string& config_path);

} // namespace dist
