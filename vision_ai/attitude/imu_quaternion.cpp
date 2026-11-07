#include <iostream>
#include <cmath>
#include <array>

// ============================================
// IMU自己位置推定用 クォータニオンライブラリ (C++17)
// ============================================

struct Vec3 {
    double x, y, z;

    Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
};

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// --------------------------------------------
// クォータニオン構造体
// q = [w, x, y, z] = q0 + q1*i + q2*j + q3*k
// --------------------------------------------
struct Quaternion {
    double w, x, y, z;

    // 単位クォータニオンで初期化
    Quaternion(double w_=1.0, double x_=0.0, double y_=0.0, double z_=0.0)
        : w(w_), x(x_), y(y_), z(z_) {}

    // ノルム（長さ）
    double norm() const {
        return std::sqrt(w*w + x*x + y*y + z*z);
    }

    // 正規化（単位クォータニオンにする）
    void normalize() {
        double n = norm();
        if (n < 1e-12) return; // ゼロ割防止
        w /= n; x /= n; y /= n; z /= n;
    }

    // 共役クォータニオン q* = [w, -x, -y, -z]
    Quaternion conjugate() const {
        return Quaternion(w, -x, -y, -z);
    }

    // クォータニオンの積（ハミルトン積）: this ⊗ other
    Quaternion operator*(const Quaternion& other) const {
        return Quaternion(
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w
        );
    }

    // スカラ倍
    Quaternion operator*(double s) const {
        return Quaternion(w * s, x * s, y * s, z * s);
    }

    Quaternion operator+(const Quaternion& other) const {
        return Quaternion(w + other.w, x + other.x, y + other.y, z + other.z);
    }
};

// --------------------------------------------
// 3次元ベクトルをクォータニオンで回転させる
// v_world = q ⊗ [0, v_body] ⊗ q*
// --------------------------------------------
inline Vec3 rotateVector(const Quaternion& q, const Vec3& v) {
    Quaternion qv(0.0, v.x, v.y, v.z);
    Quaternion qv_rot = q * qv * q.conjugate();
    return {qv_rot.x, qv_rot.y, qv_rot.z};
}

// --------------------------------------------
// クォータニオンから回転行列(3x3)を生成
// オプション: 行列形式で変換したい場合に使用
// --------------------------------------------
inline std::array<std::array<double, 3>, 3> toRotationMatrix(const Quaternion& q) {
    double w = q.w, x = q.x, y = q.y, z = q.z;
    std::array<std::array<double, 3>, 3> R = {{
        {{w*w + x*x - y*y - z*z, 2*(x*y - w*z),     2*(x*z + w*y)}},
        {{2*(x*y + w*z),         w*w - x*x + y*y - z*z, 2*(y*z - w*x)}},
        {{2*(x*z - w*y),         2*(y*z + w*x),     w*w - x*x - y*y + z*z}}
    }};
    return R;
}

// --------------------------------------------
// IMU状態（姿勢・速度・位置）
// --------------------------------------------
struct IMUState {
    Quaternion q;      // 姿勢（ボディ→ワールド変）
    Vec3 vel;          // 世界座標系での速度 [m/s]
    Vec3 pos;          // 世界座標系での位置 [m]
    double timestamp;  // 現在の時刻 [s]

    IMUState() : q(1,0,0,0), vel{0,0,0}, pos{0,0,0}, timestamp(0.0) {}
};

// --------------------------------------------
// IMU更新関数
// gyro:  角速度 [rad/s] （ボディ座標系、センサー読み値）
// accel: 加速度 [m/s²] （ボディ座標系、センサー読み値、重力含む）
// dt:    サンプリング時間 [s]
// gravity: 世界座標系での重力加速度 [m/s²] （通常 z軸負方向に -9.80665）
// --------------------------------------------
void updateIMU(IMUState& state, 
               const Vec3& gyro, 
               const Vec3& accel, 
               double dt,
               const Vec3& gravity = {0, 0, -9.80665}) 
{
    // ========================================
    // 1. 姿勢更新（クォータニオンの時間微分をオイラー積分）
    // dq/dt = 0.5 * Ω(ω) * q
    // q_{k+1} = q_k + dt * 0.5 * q ⊗ [0, ω]
    // ========================================

    // 角速度をクォータニオン形式にして積分
    Quaternion omega_q(0.0, gyro.x, gyro.y, gyro.z);
    Quaternion dq = state.q * omega_q * 0.5;
    state.q = state.q + dq * dt;
    state.q.normalize(); // 数値誤差によりノルムがずれるので正規化が必須

    // ========================================
    // 2. 加速度をボディ座標系から世界座標系へ変換
    // ========================================
    Vec3 accel_world = rotateVector(state.q, accel);

    // ========================================
    // 3. 重力補正（世界座標系で重力分を除去）
    // ========================================
    Vec3 accel_corrected = accel_world - gravity;

    // ========================================
    // 4. 速度・位置の更新（数値積分）
    // v_{k+1} = v_k + a * dt
    // p_{k+1} = p_k + v_k * dt + 0.5 * a * dt^2
    // ========================================
    state.vel = state.vel + accel_corrected * dt;
    state.pos = state.pos + state.vel * dt + accel_corrected * (0.5 * dt * dt);
    state.timestamp += dt;
}

// --------------------------------------------
// テスト用メイン関数
// --------------------------------------------
int main() {
    std::cout << std::fixed;
    std::cout.precision(6);

    // 初期状態（原点、静止、単位姿勢）
    IMUState state;

    double dt = 0.01; // 100Hzサンプリング

    // ========================================
    // テストケース1: 静止状態（重力のみ）
    // 加速度計は上方向に重力を感じるはず -> 補正後はゼロ
    // ========================================
    std::cout << "=== テスト1: 静止状態（重力のみ） ===" << std::endl;
    Vec3 gyro1 = {0, 0, 0};           // 角速度ゼロ
    Vec3 accel1 = {0, 0, 9.80665};    // ボディ座標系で上向きに重力を感じる

    for (int i = 0; i < 100; ++i) {
        updateIMU(state, gyro1, accel1, dt);
    }

    std::cout << "時刻: " << state.timestamp << " s" << std::endl;
    std::cout << "姿勢 q: [" << state.q.w << ", " << state.q.x << ", " 
              << state.q.y << ", " << state.q.z << "]" << std::endl;
    std::cout << "位置 p: [" << state.pos.x << ", " << state.pos.y << ", " << state.pos.z << "]" << std::endl;
    std::cout << "速度 v: [" << state.vel.x << ", " << state.vel.y << ", " << state.vel.z << "]" << std::endl;
    std::cout << std::endl;

    // ========================================
    // テストケース2: Z軸周りに90度/sで回転しながら、X方向に1m/s²加速
    // ========================================
    std::cout << "=== テスト2: Z軸回転 + X方向加速 ===" << std::endl;
    state = IMUState(); // リセット

    Vec3 gyro2 = {0, 0, M_PI / 2.0}; // 90度/s = π/2 rad/s
    Vec3 accel2 = {1.0, 0, 9.80665}; // ボディX方向に1m/s²上向きに重力

    for (int i = 0; i < 100; ++i) {
        updateIMU(state, gyro2, accel2, dt);
    }

    std::cout << "時刻: " << state.timestamp << " s" << std::endl;
    std::cout << "姿勢 q: [" << state.q.w << ", " << state.q.x << ", " 
              << state.q.y << ", " << state.q.z << "]" << std::endl;
    std::cout << "位置 p: [" << state.pos.x << ", " << state.pos.y << ", " << state.pos.z << "]" << std::endl;
    std::cout << "速度 v: [" << state.vel.x << ", " << state.vel.y << ", " << state.vel.z << "]" << std::endl;
    std::cout << std::endl;

    // ========================================
    // テストケース3: クォータニオン積の非可換性確認
    // ========================================
    std::cout << "=== テスト3: クォータニオン積の非可換性 ===" << std::endl;
    Quaternion p(1, 2, 0, 0);
    Quaternion q(0, 1, 1, 0);
    Quaternion pq = p * q;
    Quaternion qp = q * p;

    std::cout << "p ⊗ q = [" << pq.w << ", " << pq.x << ", " << pq.y << ", " << pq.z << "]" << std::endl;
    std::cout << "q ⊗ p = [" << qp.w << ", " << qp.x << ", " << qp.y << ", " << qp.z << "]" << std::endl;
    std::cout << "一致するか: " << (pq.w == qp.w && pq.x == qp.x && pq.y == qp.y && pq.z == qp.z ? "はい" : "いいえ") << std::endl;
    std::cout << std::endl;

    // ========================================
    // テストース4: 回行列の確認
    // ========================================
    std::cout << "=== テスト4: 回転行列 ===" << std::endl;
    Quaternion q_rot(std::cos(M_PI/4), 0, 0, std::sin(M_PI/4)); // Z軸周り90度回転
    auto R = toRotationMatrix(q_rot);
    std::cout << "Z軸90度回転行列:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << R[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
