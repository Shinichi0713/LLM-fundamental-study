#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <limits>

// LASlib のヘッダー (ご使用のライブラリ環境に合わせてインクルード)
#include "lasreader.hpp"
#include "laswriter.hpp"

// 3D 座標構造体
struct Point3D {
    double x, y, z;
};

// 色構造体 (16-bit RGB: 0 - 65535)
struct ColorRGB {
    unsigned short r, g, b;
};

// 地形 DEM（デジタル標高モデル）構造体
struct DEM {
    std::vector<double> grid;
    int cols;
    int rows;
    double x_min, x_max;
    double y_min, y_max;
    double grid_size;

    // グリッド位置から標高値を取得
    double get_elevation(double x, double y) const {
        if (x < x_min || x >= x_max || y < y_min || y >= y_max) {
            return -9999.0; // 範囲外
        }
        int col = static_cast<int>((x - x_min) / grid_size);
        int row = static_cast<int>((y - y_min) / grid_size);
        
        col = std::clamp(col, 0, cols - 1);
        row = std::clamp(row, 0, rows - 1);

        return grid[row * cols + col];
    }
};

// -----------------------------------------------------------------------------
// 1. 点群データから簡単な DEM (最近傍標高マップ) を構築
// -----------------------------------------------------------------------------
DEM create_dem(const std::vector<Point3D>& terrain_points, double grid_size) {
    DEM dem;
    dem.grid_size = grid_size;
    dem.x_min = std::numeric_limits<double>::max();
    dem.x_max = std::numeric_limits<double>::lowest();
    dem.y_min = std::numeric_limits<double>::max();
    dem.y_max = std::numeric_limits<double>::lowest();

    for (const auto& p : terrain_points) {
        dem.x_min = std::min(dem.x_min, p.x);
        dem.x_max = std::max(dem.x_max, p.x);
        dem.y_min = std::min(dem.y_min, p.y);
        dem.y_max = std::max(dem.y_max, p.y);
    }

    dem.cols = static_cast<int>(std::ceil((dem.x_max - dem.x_min) / grid_size));
    dem.rows = static_cast<int>(std::ceil((dem.y_max - dem.y_min) / grid_size));
    dem.grid.assign(dem.cols * dem.rows, -9999.0);

    // 最大標高値でグリッドを更新 (簡易Grid化)
    for (const auto& p : terrain_points) {
        int col = static_cast<int>((p.x - dem.x_min) / grid_size);
        int row = static_cast<int>((p.y - dem.y_min) / grid_size);
        
        col = std::clamp(col, 0, dem.cols - 1);
        row = std::clamp(row, 0, dem.rows - 1);

        size_t idx = row * dem.cols + col;
        dem.grid[idx] = std::max(dem.grid[idx], p.z);
    }

    return dem;
}

// -----------------------------------------------------------------------------
// 2. 視線解析 (Line of Sight)
// -----------------------------------------------------------------------------
bool is_visible(const Point3D& observer, const Point3D& target, const DEM& dem, 
                double observer_offset = 1.7, int sample_num = 100) {
    
    // 監視者の視点高さ (目の高さ)
    double obs_x = observer.x;
    double obs_y = observer.y;
    double obs_z = observer.z + observer_offset;

    double tgt_x = target.x;
    double tgt_y = target.y;
    double tgt_z = target.z;

    // 視線上のサンプル点を線形補間してチェック
    for (int i = 1; i < sample_num; ++i) {
        double ratio = static_cast<double>(i) / sample_num;
        
        double sample_x = obs_x + ratio * (tgt_x - obs_x);
        double sample_y = obs_y + ratio * (tgt_y - obs_y);
        double sample_z = obs_z + ratio * (tgt_z - obs_z);

        // その地点の地形標高を取得
        double terrain_z = dem.get_elevation(sample_x, sample_y);

        // 視線の高度が地形標高を下回っていれば「遮蔽あり (不可視)」
        if (terrain_z > -9000.0 && sample_z < terrain_z) {
            return false;
        }
    }
    return true; // 遮蔽物なし (可視)
}

// -----------------------------------------------------------------------------
// 3. Viewshed解析と結果のLAS書き出し
// -----------------------------------------------------------------------------
void run_viewshed_analysis(const std::string& input_laz_path,
                           const std::vector<Point3D>& observers,
                           const std::vector<Point3D>& waypoints,
                           const std::string& output_las_path) {

    // A. LAZ点群の読み込み (LASlib使用例)
    LASreadOpener lasreadopener;
    lasreadopener.set_file_name(input_laz_path.c_str());
    LASreader* lasreader = lasreadopener.open();

    if (!lasreader) {
        std::cerr << "エラー: LAZファイルを開けませんでした: " << input_laz_path << std::endl;
        return;
    }

    std::vector<Point3D> terrain_points;
    terrain_points.reserve(lasreader->npoints);

    while (lasreader->read_point()) {
        terrain_points.push_back({lasreader->point.get_x(), 
                                  lasreader->point.get_y(), 
                                  lasreader->point.get_z()});
    }
    lasreader->close();
    delete lasreader;

    std::cout << "読み込み完了: " << terrain_points.size() << " ポイント" << std::endl;

    // B. 地形 DEM の生成 (解像度: 0.5m)
    double grid_size = 0.5;
    DEM dem = create_dem(terrain_points, grid_size);

    // C. 可視性判定およびカラー割り当て
    std::vector<ColorRGB> colors;
    int visible_count = 0;

    for (const auto& wp : waypoints) {
        bool visible = false;
        for (const auto& obs : observers) {
            if (is_visible(obs, wp, dem, 1.7)) {
                visible = true;
                break;
            }
        }

        if (visible) {
            colors.push_back({65535, 0, 0}); // 赤色 (可視)
            visible_count++;
        } else {
            colors.push_back({0, 0, 65535}); // 青色 (不可視)
        }
    }

    // D. ウェイポイントデータと色情報を書き出し (LASlib)
    LASheader header;
    header.x_scale_factor = 0.01;
    header.y_scale_factor = 0.01;
    header.z_scale_factor = 0.01;
    header.point_data_format = 3; // RGB対応フォーマット
    header.point_data_record_length = 34;

    LASwriteOpener laswriteopener;
    laswriteopener.set_file_name(output_las_path.c_str());
    LASwriter* laswriter = laswriteopener.open(&header);

    LASpoint point;
    point.init(&header, header.point_data_format, header.point_data_record_length, nullptr);

    for (size_t i = 0; i < waypoints.size(); ++i) {
        point.set_x(waypoints[i].x);
        point.set_y(waypoints[i].y);
        point.set_z(waypoints[i].z);
        
        point.rgb[0] = colors[i].r;
        point.rgb[1] = colors[i].g;
        point.rgb[2] = colors[i].b;

        laswriter->write_point(&point);
        laswriter->p_count++;
    }

    laswriter->close();
    delete laswriter;

    std::cout << "解析完了: 全 " << waypoints.size() << " ポイント中 "
              << visible_count << " ポイントが可視 (赤) です。" << std::endl;
    std::cout << "結果出力先: " << output_las_path << std::endl;
}

// -----------------------------------------------------------------------------
// メイン関数
// -----------------------------------------------------------------------------
int main() {
    std::string input_laz = "terrain_data.laz";
    std::string output_las = "colored_waypoints.las";

    // 監視者座標リスト [X, Y, Z]
    std::vector<Point3D> observers = {
        {100.0, 150.0, 10.5},
        {200.0, 250.0, 12.0}
    };

    // UAVウェイポイント座標リスト [X, Y, Z]
    std::vector<Point3D> waypoints = {
        {105.0, 155.0, 30.0},
        {150.0, 200.0, 25.0},
        {180.0, 220.0, 15.0},
        {300.0, 300.0, 40.0}
    };

    // run_viewshed_analysis(input_laz, observers, waypoints, output_las);

    return 0;
}