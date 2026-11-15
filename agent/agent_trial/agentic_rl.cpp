// =============================================================================
// Agentic RL - Minimal C++ Implementation (C++17, PyTorch-free)
// =============================================================================
//
// Task: "Reach Target Number"
//   Start from 1, reach exactly 10 within 10 steps.
//   Actions: +1, +2, x2
//
// Algorithm: Actor-Critic (A2C) with LSTM-based reasoning state
//   - LSTMCell: maintains a "reasoning hidden state" across steps
//   - Policy head: decides action
//   - Value head: estimates state value for bootstrapping
//   - Adam optimizer for all parameters
//
// Build:
//   g++ -std=c++17 -O2 -o agentic_rl agentic_rl.cpp
//
// Output:
//   - Console: training progress + evaluation trajectories
//   - training_curve.csv: episode reward / length / success flag
//
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include <iomanip>
#include <tuple>
#include <cassert>
#include <sstream>

// =============================================================================
// Global RNG
// =============================================================================
std::mt19937 g_rng(42);

// =============================================================================
// Math Utilities
// =============================================================================

double sigmoid(double x) {
    if (x > 20.0)  return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<double> softmax(const std::vector<double>& x) {
    double max_val = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(x[i] - max_val);
        sum += out[i];
    }
    for (size_t i = 0; i < x.size(); ++i) out[i] /= sum;
    return out;
}

int sample_categorical(const std::vector<double>& probs) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(g_rng);
}

int argmax(const std::vector<double>& x) {
    return static_cast<int>(std::max_element(x.begin(), x.end()) - x.begin());
}

// =============================================================================
// Linear Layer (fully connected)
// =============================================================================

class Linear {
public:
    int in_features;
    int out_features;
    std::vector<std::vector<double>> weight;
    std::vector<double> bias;

    std::vector<std::vector<double>> grad_w;
    std::vector<double> grad_b;

    std::vector<std::vector<double>> m_w;
    std::vector<double> m_b;
    std::vector<std::vector<double>> v_w;
    std::vector<double> v_b;

    Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
        weight.assign(out_features, std::vector<double>(in_features, 0.0));
        bias.assign(out_features, 0.0);
        grad_w.assign(out_features, std::vector<double>(in_features, 0.0));
        grad_b.assign(out_features, 0.0);
        m_w.assign(out_features, std::vector<double>(in_features, 0.0));
        m_b.assign(out_features, 0.0);
        v_w.assign(out_features, std::vector<double>(in_features, 0.0));
        v_b.assign(out_features, 0.0);
        initialize_weights();
    }

    void initialize_weights() {
        double scale = std::sqrt(2.0 / (in_features + out_features));
        std::normal_distribution<double> dist(0.0, scale);
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weight[i][j] = dist(g_rng);
            }
            bias[i] = 0.0;
        }
    }

    std::vector<double> forward(const std::vector<double>& x) const {
        assert(static_cast<int>(x.size()) == in_features);
        std::vector<double> y(out_features, 0.0);
        for (int i = 0; i < out_features; ++i) {
            y[i] = bias[i];
            for (int j = 0; j < in_features; ++j) {
                y[i] += weight[i][j] * x[j];
            }
        }
        return y;
    }

    std::vector<double> backward(const std::vector<double>& grad_output,
                                 const std::vector<double>& input) {
        assert(static_cast<int>(grad_output.size()) == out_features);
        assert(static_cast<int>(input.size()) == in_features);
        std::vector<double> grad_input(in_features, 0.0);
        for (int i = 0; i < out_features; ++i) {
            grad_b[i] += grad_output[i];
            for (int j = 0; j < in_features; ++j) {
                grad_w[i][j] += grad_output[i] * input[j];
                grad_input[j] += grad_output[i] * weight[i][j];
            }
        }
        return grad_input;
    }

    void zero_grad() {
        for (int i = 0; i < out_features; ++i) {
            std::fill(grad_w[i].begin(), grad_w[i].end(), 0.0);
        }
        std::fill(grad_b.begin(), grad_b.end(), 0.0);
    }

    void step(double lr, int timestep, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8) {
        double bias_corr1 = 1.0 - std::pow(beta1, timestep);
        double bias_corr2 = 1.0 - std::pow(beta2, timestep);
        for (int i = 0; i < out_features; ++i) {
            m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * grad_b[i];
            v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * grad_b[i] * grad_b[i];
            double m_hat = m_b[i] / bias_corr1;
            double v_hat = v_b[i] / bias_corr2;
            bias[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);

            for (int j = 0; j < in_features; ++j) {
                m_w[i][j] = beta1 * m_w[i][j] + (1.0 - beta1) * grad_w[i][j];
                v_w[i][j] = beta2 * v_w[i][j] + (1.0 - beta2) * grad_w[i][j] * grad_w[i][j];
                double m_hat_w = m_w[i][j] / bias_corr1;
                double v_hat_w = v_w[i][j] / bias_corr2;
                weight[i][j] -= lr * m_hat_w / (std::sqrt(v_hat_w) + eps);
            }
        }
    }
};

// =============================================================================
// ReLU Activation
// =============================================================================

class ReLU {
public:
    std::vector<double> forward(const std::vector<double>& x) const {
        std::vector<double> out(x.size());
        for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] > 0.0 ? x[i] : 0.0;
        return out;
    }

    std::vector<double> backward(const std::vector<double>& grad_output,
                                 const std::vector<double>& input) const {
        std::vector<double> grad_input(grad_output.size());
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_input[i] = grad_output[i] * (input[i] > 0.0 ? 1.0 : 0.0);
        }
        return grad_input;
    }
};

// =============================================================================
// LSTM Cell (single step, batch_size = 1)
// =============================================================================

class LSTMCell {
public:
    int input_size;
    int hidden_size;
    int total_in;

    std::vector<std::vector<double>> W;
    std::vector<double> b;

    std::vector<std::vector<double>> grad_W;
    std::vector<double> grad_b;

    std::vector<std::vector<double>> m_W;
    std::vector<double> m_b;
    std::vector<std::vector<double>> v_W;
    std::vector<double> v_b;

    struct Cache {
        std::vector<double> x;
        std::vector<double> h_prev;
        std::vector<double> c_prev;
        std::vector<double> z;
        std::vector<double> i, f, g, o;
        std::vector<double> c;
        std::vector<double> h;
    };
    std::vector<Cache> caches;

    LSTMCell(int in_sz, int hid_sz) : input_size(in_sz), hidden_size(hid_sz) {
        total_in = input_size + hidden_size;
        int gate_size = 4 * hidden_size;
        W.assign(gate_size, std::vector<double>(total_in, 0.0));
        b.assign(gate_size, 0.0);
        grad_W.assign(gate_size, std::vector<double>(total_in, 0.0));
        grad_b.assign(gate_size, 0.0);
        m_W.assign(gate_size, std::vector<double>(total_in, 0.0));
        m_b.assign(gate_size, 0.0);
        v_W.assign(gate_size, std::vector<double>(total_in, 0.0));
        v_b.assign(gate_size, 0.0);
        initialize_weights();
    }

    void initialize_weights() {
        double scale = std::sqrt(2.0 / (total_in + 4 * hidden_size));
        std::normal_distribution<double> dist(0.0, scale);
        for (int i = 0; i < 4 * hidden_size; ++i) {
            for (int j = 0; j < total_in; ++j) {
                W[i][j] = dist(g_rng);
            }
            b[i] = 0.0;
            if (i >= hidden_size && i < 2 * hidden_size) {
                b[i] = 1.0;
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& input,
                                const std::vector<double>& h_prev,
                                const std::vector<double>& c_prev,
                                std::vector<double>& c_out) {
        assert(static_cast<int>(input.size()) == input_size);
        assert(static_cast<int>(h_prev.size()) == hidden_size);
        assert(static_cast<int>(c_prev.size()) == hidden_size);

        std::vector<double> x(total_in);
        for (int j = 0; j < input_size; ++j) x[j] = input[j];
        for (int j = 0; j < hidden_size; ++j) x[input_size + j] = h_prev[j];

        std::vector<double> z(4 * hidden_size, 0.0);
        for (int i = 0; i < 4 * hidden_size; ++i) {
            z[i] = b[i];
            for (int j = 0; j < total_in; ++j) {
                z[i] += W[i][j] * x[j];
            }
        }

        std::vector<double> i_gate(hidden_size);
        std::vector<double> f_gate(hidden_size);
        std::vector<double> g_gate(hidden_size);
        std::vector<double> o_gate(hidden_size);
        for (int j = 0; j < hidden_size; ++j) {
            i_gate[j] = sigmoid(z[j]);
            f_gate[j] = sigmoid(z[hidden_size + j]);
            g_gate[j] = std::tanh(z[2 * hidden_size + j]);
            o_gate[j] = sigmoid(z[3 * hidden_size + j]);
        }

        c_out.resize(hidden_size);
        for (int j = 0; j < hidden_size; ++j) {
            c_out[j] = f_gate[j] * c_prev[j] + i_gate[j] * g_gate[j];
        }

        std::vector<double> h(hidden_size);
        for (int j = 0; j < hidden_size; ++j) {
            h[j] = o_gate[j] * std::tanh(c_out[j]);
        }

        Cache cache;
        cache.x = x;
        cache.h_prev = h_prev;
        cache.c_prev = c_prev;
        cache.z = z;
        cache.i = i_gate;
        cache.f = f_gate;
        cache.g = g_gate;
        cache.o = o_gate;
        cache.c = c_out;
        cache.h = h;
        caches.push_back(cache);

        return h;
    }

    void backward(const std::vector<std::vector<double>>& dh_list,
                  std::vector<std::vector<double>>& grad_inputs,
                  std::vector<double>& grad_h0,
                  std::vector<double>& grad_c0) {
        int T = static_cast<int>(caches.size());
        assert(T == static_cast<int>(dh_list.size()));
        int H = hidden_size;

        grad_inputs.assign(T, std::vector<double>(input_size, 0.0));
        grad_h0.assign(H, 0.0);
        grad_c0.assign(H, 0.0);

        std::vector<double> dc(H, 0.0);
        std::vector<double> dh_next(H, 0.0);

        for (int t = T - 1; t >= 0; --t) {
            const Cache& cache = caches[t];

            std::vector<double> dh(H, 0.0);
            for (int j = 0; j < H; ++j) {
                dh[j] = dh_list[t][j] + dh_next[j];
            }

            std::vector<double> tanh_c(H);
            for (int j = 0; j < H; ++j) tanh_c[j] = std::tanh(cache.c[j]);

            std::vector<double> dzo(H);
            for (int j = 0; j < H; ++j) {
                double do_gate = dh[j] * tanh_c[j];
                dzo[j] = do_gate * cache.o[j] * (1.0 - cache.o[j]);
            }

            std::vector<double> dct(H);
            for (int j = 0; j < H; ++j) {
                dct[j] = dc[j] + dh[j] * cache.o[j] * (1.0 - tanh_c[j] * tanh_c[j]);
            }

            std::vector<double> dzi(H);
            for (int j = 0; j < H; ++j) {
                dzi[j] = dct[j] * cache.g[j] * cache.i[j] * (1.0 - cache.i[j]);
            }

            std::vector<double> dzf(H);
            for (int j = 0; j < H; ++j) {
                dzf[j] = dct[j] * cache.c_prev[j] * cache.f[j] * (1.0 - cache.f[j]);
            }

            std::vector<double> dzg(H);
            for (int j = 0; j < H; ++j) {
                dzg[j] = dct[j] * cache.i[j] * (1.0 - cache.g[j] * cache.g[j]);
            }

            std::vector<double> dz(4 * H);
            for (int j = 0; j < H; ++j) {
                dz[j] = dzi[j];
                dz[H + j] = dzf[j];
                dz[2 * H + j] = dzg[j];
                dz[3 * H + j] = dzo[j];
            }

            for (int i = 0; i < 4 * H; ++i) {
                grad_b[i] += dz[i];
                for (int j = 0; j < total_in; ++j) {
                    grad_W[i][j] += dz[i] * cache.x[j];
                }
            }

            std::vector<double> dx(total_in, 0.0);
            for (int j = 0; j < total_in; ++j) {
                for (int i = 0; i < 4 * H; ++i) {
                    dx[j] += dz[i] * W[i][j];
                }
            }

            for (int j = 0; j < input_size; ++j) grad_inputs[t][j] = dx[j];
            if (t > 0) {
                for (int j = 0; j < H; ++j) dh_next[j] = dx[input_size + j];
            } else {
                for (int j = 0; j < H; ++j) grad_h0[j] = dx[input_size + j];
            }

            if (t > 0) {
                for (int j = 0; j < H; ++j) dc[j] = dct[j] * cache.f[j];
            } else {
                for (int j = 0; j < H; ++j) grad_c0[j] = dct[j] * cache.f[j];
            }
        }

        caches.clear();
    }

    void zero_grad() {
        for (size_t i = 0; i < grad_W.size(); ++i) {
            std::fill(grad_W[i].begin(), grad_W[i].end(), 0.0);
        }
        std::fill(grad_b.begin(), grad_b.end(), 0.0);
    }

    void step(double lr, int timestep, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8) {
        double bias_corr1 = 1.0 - std::pow(beta1, timestep);
        double bias_corr2 = 1.0 - std::pow(beta2, timestep);
        int gate_size = 4 * hidden_size;
        for (int i = 0; i < gate_size; ++i) {
            m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * grad_b[i];
            v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * grad_b[i] * grad_b[i];
            double m_hat = m_b[i] / bias_corr1;
            double v_hat = v_b[i] / bias_corr2;
            b[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);

            for (int j = 0; j < total_in; ++j) {
                m_W[i][j] = beta1 * m_W[i][j] + (1.0 - beta1) * grad_W[i][j];
                v_W[i][j] = beta2 * v_W[i][j] + (1.0 - beta2) * grad_W[i][j] * grad_W[i][j];
                double m_hat_w = m_W[i][j] / bias_corr1;
                double v_hat_w = v_W[i][j] / bias_corr2;
                W[i][j] -= lr * m_hat_w / (std::sqrt(v_hat_w) + eps);
            }
        }
    }

    void clear_caches() {
        caches.clear();
    }
};

// =============================================================================
// Environment: Reach Target Number
// =============================================================================

struct StepResult {
    std::vector<double> state;
    double reward;
    bool done;
};

class Environment {
public:
    static constexpr int TARGET = 10;
    static constexpr int MAX_STEPS = 10;

    int current = 1;
    int steps = 0;
    int last_action = -1;

    Environment() { reset(); }

    std::vector<double> reset() {
        current = 1;
        steps = 0;
        last_action = -1;
        return get_state();
    }

    std::vector<double> get_state() const {
        std::vector<double> last_action_onehot(3, 0.0);
        if (last_action >= 0) last_action_onehot[last_action] = 1.0;
        return {
            current / static_cast<double>(TARGET),
            steps / static_cast<double>(MAX_STEPS),
            last_action_onehot[0],
            last_action_onehot[1],
            last_action_onehot[2],
            1.0
        };
    }

    StepResult step(int action) {
        steps++;
        last_action = action;

        if (action == 0) {
            current += 1;
        } else if (action == 1) {
            current += 2;
        } else if (action == 2) {
            current *= 2;
        }

        double reward = -0.05;
        bool done = false;

        if (current == TARGET) {
            reward = 10.0;
            done = true;
        } else if (current > TARGET) {
            reward = -5.0;
            done = true;
        } else if (steps >= MAX_STEPS) {
            reward = -5.0;
            done = true;
        }

        return {get_state(), reward, done};
    }

    static std::string action_name(int action) {
        if (action == 0) return "+1";
        if (action == 1) return "+2";
        return "x2";
    }
};

// =============================================================================
// Transition & Trajectory
// =============================================================================

struct Transition {
    std::vector<double> state;
    int action;
    double value;
    std::vector<double> policy_probs;
    double reward;
};

struct Trajectory {
    std::vector<Transition> transitions;
};

// =============================================================================
// Agent: LSTM-based Actor-Critic
// =============================================================================

class Agent {
public:
    Linear encoder;
    ReLU relu;
    LSTMCell lstm;
    Linear policy_head;
    Linear value_head;

    std::vector<double> h_prev;
    std::vector<double> c_prev;

    std::vector<std::vector<double>> encoded_history;
    std::vector<std::vector<double>> h_history;

    int action_dim;

    Agent(int state_dim, int act_dim, int hidden_dim)
        : encoder(state_dim, 64),
          lstm(64, hidden_dim),
          policy_head(hidden_dim, act_dim),
          value_head(hidden_dim, 1),
          action_dim(act_dim) {
        h_prev.assign(hidden_dim, 0.0);
        c_prev.assign(hidden_dim, 0.0);
    }

    void reset_hidden() {
        std::fill(h_prev.begin(), h_prev.end(), 0.0);
        std::fill(c_prev.begin(), c_prev.end(), 0.0);
        encoded_history.clear();
        h_history.clear();
        lstm.clear_caches();
    }

    std::pair<std::vector<double>, double> forward(const std::vector<double>& state) {
        std::vector<double> encoded = encoder.forward(state);
        encoded = relu.forward(encoded);
        encoded_history.push_back(encoded);

        std::vector<double> c_new;
        std::vector<double> h = lstm.forward(encoded, h_prev, c_prev, c_new);
        h_prev = h;
        c_prev = c_new;
        h_history.push_back(h);

        std::vector<double> logits = policy_head.forward(h);
        std::vector<double> value_vec = value_head.forward(h);

        return {logits, value_vec[0]};
    }

    void backward(const Trajectory& traj, const std::vector<double>& returns) {
        int T = static_cast<int>(traj.transitions.size());
        int H = lstm.hidden_size;

        std::vector<double> advantages(T);
        for (int t = 0; t < T; ++t) {
            advantages[t] = returns[t] - traj.transitions[t].value;
        }

        std::vector<std::vector<double>> dh_list(T, std::vector<double>(H, 0.0));

        for (int t = 0; t < T; ++t) {
            std::vector<double> dlogits(action_dim, 0.0);
            int a = traj.transitions[t].action;
            for (int i = 0; i < action_dim; ++i) {
                double pi = traj.transitions[t].policy_probs[i];
                if (i == a) dlogits[i] = -advantages[t] * (1.0 - pi);
                else dlogits[i] = -advantages[t] * (-pi);
            }
            std::vector<double> dh_p = policy_head.backward(dlogits, h_history[t]);
            for (int j = 0; j < H; ++j) dh_list[t][j] += dh_p[j];

            double dvalue = 2.0 * (traj.transitions[t].value - returns[t]) / T;
            std::vector<double> dvalue_vec = {dvalue};
            std::vector<double> dh_v = value_head.backward(dvalue_vec, h_history[t]);
            for (int j = 0; j < H; ++j) dh_list[t][j] += dh_v[j];
        }

        std::vector<std::vector<double>> grad_encoded;
        std::vector<double> grad_h0, grad_c0;
        lstm.backward(dh_list, grad_encoded, grad_h0, grad_c0);

        for (int t = T - 1; t >= 0; --t) {
            std::vector<double> de = relu.backward(grad_encoded[t], encoded_history[t]);
            encoder.backward(de, traj.transitions[t].state);
        }
    }

    void zero_grad() {
        encoder.zero_grad();
        lstm.zero_grad();
        policy_head.zero_grad();
        value_head.zero_grad();
    }

    void step(double lr, int timestep) {
        encoder.step(lr, timestep);
        lstm.step(lr, timestep);
        policy_head.step(lr, timestep);
        value_head.step(lr, timestep);
    }
};

// =============================================================================
// Trainer: A2C
// =============================================================================

class Trainer {
public:
    Agent& agent;
    Environment& env;
    double gamma;
    double lr;
    int global_step;

    Trainer(Agent& a, Environment& e, double g, double l)
        : agent(a), env(e), gamma(g), lr(l), global_step(0) {}

    Trajectory collect_trajectory() {
        Trajectory traj;
        std::vector<double> state = env.reset();
        agent.reset_hidden();

        for (int step = 0; step < Environment::MAX_STEPS; ++step) {
            auto result = agent.forward(state);
            std::vector<double> logits = result.first;
            double value = result.second;
            std::vector<double> probs = softmax(logits);
            int action = sample_categorical(probs);

            Transition trans;
            trans.state = state;
            trans.action = action;
            trans.value = value;
            trans.policy_probs = probs;

            StepResult sr = env.step(action);
            trans.reward = sr.reward;
            traj.transitions.push_back(trans);

            state = sr.state;
            if (sr.done) break;
        }

        return traj;
    }

    std::vector<double> compute_returns(const Trajectory& traj) {
        int T = static_cast<int>(traj.transitions.size());
        std::vector<double> returns(T, 0.0);
        double G = 0.0;
        for (int t = T - 1; t >= 0; --t) {
            G = traj.transitions[t].reward + gamma * G;
            returns[t] = G;
        }
        return returns;
    }
};

// =============================================================================
// Evaluation
// =============================================================================

void evaluate_agent(Agent& agent, Environment& env, int num_episodes = 10) {
    int successes = 0;
    for (int ep = 0; ep < num_episodes; ++ep) {
        std::vector<double> state = env.reset();
        agent.reset_hidden();

        std::vector<std::string> traj_lines;
        bool done = false;

        for (int step = 0; step < Environment::MAX_STEPS; ++step) {
            auto result = agent.forward(state);
            std::vector<double> logits = result.first;
            std::vector<double> probs = softmax(logits);
            int action = argmax(probs);

            int old_val = env.current;
            StepResult sr = env.step(action);
            int new_val = env.current;

            std::string line = std::to_string(old_val) + " --" + Environment::action_name(action) + "--> " + std::to_string(new_val) + " (probs: [";
            for (int i = 0; i < 3; ++i) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << probs[i];
                line += oss.str();
                if (i < 2) line += ", ";
            }
            line += "])";
            traj_lines.push_back(line);

            state = sr.state;
            done = sr.done;
            if (done) break;
        }

        bool success = (env.current == Environment::TARGET);
        if (success) successes++;

        std::string status = success ? "SUCCESS" : "FAILED";
        std::cout << "\nEpisode " << (ep + 1) << " [" << status << "] | Final: " << env.current
                  << " | Steps: " << traj_lines.size() << "\n";
        for (const auto& line : traj_lines) {
            std::cout << "  " << line << "\n";
        }
    }
    std::cout << "\nSuccess Rate: " << successes << "/" << num_episodes
              << " (" << (100 * successes / num_episodes) << "%)\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "Agentic RL - Minimal C++ Implementation (PyTorch-free)\n";
    std::cout << "============================================================\n";
    std::cout << "\nTask: Start from 1, reach exactly 10 using +1, +2, or x2\n";
    std::cout << "Algorithm: Actor-Critic (A2C) with LSTM reasoning state\n";
    std::cout << "Optimizer: Adam (beta1=0.9, beta2=0.999)\n";
    std::cout << "------------------------------------------------------------\n";

    Environment env;
    Agent agent(5, 3, 128);
    Trainer trainer(agent, env, 0.99, 3e-3);

    std::cout << "\nAgent Architecture:\n";
    std::cout << "  Encoder:   Linear(5 -> 64) + ReLU\n";
    std::cout << "  LSTM:      LSTMCell(64 -> 128)\n";
    std::cout << "  Policy:    Linear(128 -> 3)\n";
    std::cout << "  Value:     Linear(128 -> 1)\n";

    int total_params = 0;
    total_params += agent.encoder.in_features * agent.encoder.out_features + agent.encoder.out_features;
    total_params += agent.lstm.total_in * (4 * agent.lstm.hidden_size) + 4 * agent.lstm.hidden_size;
    total_params += agent.policy_head.in_features * agent.policy_head.out_features + agent.policy_head.out_features;
    total_params += agent.value_head.in_features * agent.value_head.out_features + agent.value_head.out_features;
    std::cout << "  Approx Parameters: " << total_params << "\n";

    std::ofstream csv("training_curve.csv");
    csv << "episode,total_reward,length,success\n";

    std::vector<double> recent_rewards;
    const int NUM_EPISODES = 2000;
    const int PRINT_EVERY = 200;

    std::cout << "\nStarting Training...\n";
    std::cout << "------------------------------------------------------------\n";

    for (int ep = 0; ep < NUM_EPISODES; ++ep) {
        Trajectory traj = trainer.collect_trajectory();
        std::vector<double> returns = trainer.compute_returns(traj);

        double total_reward = 0.0;
        for (const auto& trans : traj.transitions) total_reward += trans.reward;
        bool success = (env.current == Environment::TARGET);

        recent_rewards.push_back(total_reward);
        if (recent_rewards.size() > PRINT_EVERY) {
            recent_rewards.erase(recent_rewards.begin());
        }

        agent.zero_grad();
        agent.backward(traj, returns);
        agent.step(trainer.lr, ++trainer.global_step);

        csv << (ep + 1) << "," << total_reward << "," << traj.transitions.size() << "," << (success ? 1 : 0) << "\n";

        if ((ep + 1) % PRINT_EVERY == 0) {
            double avg_reward = std::accumulate(recent_rewards.begin(), recent_rewards.end(), 0.0)
                                / recent_rewards.size();
            int recent_success = 0;
            for (double r : recent_rewards) if (r > 0) recent_success++;
            std::cout << "Episode " << std::setw(4) << (ep + 1) << "/" << NUM_EPISODES
                      << " | Avg Reward: " << std::fixed << std::setprecision(2) << avg_reward
                      << " | Success: " << recent_success << "/" << recent_rewards.size() << "\n";
        }
    }

    csv.close();
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Training complete. Curve saved to: training_curve.csv\n";

    std::cout << "\n============================================================\n";
    std::cout << "EVALUATION: Agentic RL Agent Reasoning Trajectories\n";
    std::cout << "============================================================\n";
    evaluate_agent(agent, env, 10);

    std::cout << "\n============================================================\n";
    std::cout << "Done!\n";
    std::cout << "============================================================\n";

    return 0;
}