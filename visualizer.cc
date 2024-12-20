#include "global_segment_map/utils/visualizer.h"
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <string>


namespace voxblox {

Visualizer::Visualizer(
    const std::vector<std::shared_ptr<MeshLayer>>& mesh_layers,
    bool* mesh_layer_updated, std::mutex* mesh_layer_mutex_ptr,
    std::vector<double> camera_position, std::vector<double> clip_distances,
    bool save_visualizer_frames)
    : mesh_layers_(mesh_layers),
      mesh_layer_updated_(CHECK_NOTNULL(mesh_layer_updated)),
      mesh_layer_mutex_ptr_(CHECK_NOTNULL(mesh_layer_mutex_ptr)),
      frame_count_(0u),
      camera_position_(camera_position),
      clip_distances_(clip_distances),
      save_visualizer_frames_(save_visualizer_frames) {}

// TODO(grinvalm): make it more efficient by only updating the
// necessary polygons and not all of them each time.
void Visualizer::visualizeMesh() {
  uint8_t n_visualizers = mesh_layers_.size();

  std::vector<std::shared_ptr<pcl::visualization::PCLVisualizer>>
      pcl_visualizers;
  std::vector<voxblox::Mesh> meshes;
  std::vector<pcl::PointCloud<pcl::PointXYZRGBA>> pointclouds;
  std::vector<pcl::PolygonMesh> polygon_meshes;

  pcl_visualizers.reserve(n_visualizers);
  meshes.resize(n_visualizers);
  pointclouds.resize(n_visualizers);
  polygon_meshes.resize(n_visualizers);

  bool refresh = false;

  for (int index = 0; index < n_visualizers; ++index) {
    // PCLVisualizer class can NOT be used across multiple threads, thus need to
    // create instances of it in the same thread they will be used in.
    std::shared_ptr<pcl::visualization::PCLVisualizer> visualizer =
        std::make_shared<pcl::visualization::PCLVisualizer>();
    std::string name = "Map " + std::to_string(index + 1);
    visualizer->setWindowName(name.c_str());
    visualizer->setBackgroundColor(255, 255, 255);
    visualizer->initCameraParameters();

    if (camera_position_.size()) {
      visualizer->setCameraPosition(
          camera_position_[0], camera_position_[1], camera_position_[2],
          camera_position_[3], camera_position_[4], camera_position_[5],
          camera_position_[6], camera_position_[7], camera_position_[8]);
    }
    if (clip_distances_.size()) {
      visualizer->setCameraClipDistances(clip_distances_[0],
                                         clip_distances_[1]);
    }

    pcl_visualizers.push_back(visualizer);
  }

  while (true) {
    for (int index = 0; index < n_visualizers; ++index) {
      constexpr int kUpdateIntervalMs = 1000;
      pcl_visualizers[index]->spinOnce(kUpdateIntervalMs);
    }
    meshes.clear();
    meshes.resize(n_visualizers);

    if (mesh_layer_mutex_ptr_->try_lock()) {
      if (*mesh_layer_updated_) {
        for (int index = 0; index < n_visualizers; index++) {
          mesh_layers_[index]->getMesh(&meshes[index]);
        }
        refresh = true;
        *mesh_layer_updated_ = false;
      }
      mesh_layer_mutex_ptr_->unlock();
    }

    if (refresh) {
      for (int index = 0; index < n_visualizers; index++) {
        pointclouds[index].points.clear();
      }

      pcl::PCLPointCloud2 pcl_pc;
      std::vector<pcl::Vertices> polygons;

      for (int index = 0; index < n_visualizers; index++) {
        size_t vert_idx = 0;
        for (const Point& vert : meshes[index].vertices) {
          pcl::PointXYZRGBA point;
          point.x = vert(0);
          point.y = vert(1);
          point.z = vert(2);

          const Color& color = meshes[index].colors[vert_idx];
          point.r = color.r;
          point.g = color.g;
          point.b = color.b;
          point.a = color.a;
          pointclouds[index].points.push_back(point);

          vert_idx++;
        }
        std::vector<pcl::PointCloud<pcl::PointXYZRGBA>>
            all_pointclouds;  // 存储所有帧的点云

        // 假设在您的可视化循环中，您已经收集了点云
        // 这里是一个示例，假设每帧的点云存储在 pointclouds 中
        for (int index = 0; index < n_visualizers; index++) {
          // 假设 pointclouds[index] 是当前帧的点云
          all_pointclouds.push_back(pointclouds[index]);
        }

        // 降采样
        pcl::PointCloud<pcl::PointXYZRGBA> combined_cloud;  // 合并后的点云
        for (const auto& cloud : all_pointclouds) {
          combined_cloud += cloud;  // 合并所有点云
        }

        // 使用 Voxel Grid 滤波器降采样
        pcl::VoxelGrid<pcl::PointXYZRGBA> voxel_filter;
        voxel_filter.setInputCloud(combined_cloud.makeShared());
        voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);  // 设置体素大小
        voxel_filter.filter(combined_cloud);            // 执行降采样

        // 保存到 PCD 文件
        if (pcl::io::savePCDFile("/home/xuyuan/pan/output.pcd",
                                 combined_cloud) == -1) {
          std::cerr << "Error saving PCD file!" << std::endl;
        } else {
          std::cout << "PCD file saved successfully!" << std::endl;
        }
        for (size_t i = 0u; i < meshes[index].indices.size(); i += 3u) {
          pcl::Vertices face;
          for (int j = 0; j < 3; j++) {
            face.vertices.push_back(meshes[index].indices.at(i + j));
          }
          polygons.push_back(face);
        }

        pcl::toPCLPointCloud2(pointclouds[index], pcl_pc);
        polygon_meshes[index].cloud = pcl_pc;
        polygon_meshes[index].polygons = polygons;
      }

      // 更新可视化窗口
      for (int index = 0; index < n_visualizers; index++) {
        // 保存mesh为PCD文件
        pcl::PointCloud<pcl::PointXYZRGBA> pointcloud;

        // 你需要从 polygon_meshes 中提取点云数据
        const pcl::PolygonMesh& mesh = polygon_meshes[index];

        // 将 polygon_meshes 转换为 pcl::PointCloud<pcl::PointXYZRGBA>
        pcl::fromPCLPointCloud2(mesh.cloud, pointcloud);

        // 设置文件保存路径，这里可以使用指定的路径
        std::string pcd_filename =
           "/home/xuyuan/pan/mesh/mesh_" + std::to_string(index) + ".pcd";

        // 保存PCD文件
        if (pcl::io::savePCDFile(pcd_filename, pointcloud) == -1) {
          std::cerr << "Error saving PCD file: " << pcd_filename << std::endl;
        } else {
          std::cout << "Saved mesh as PCD file: " << pcd_filename << std::endl;
        }
        pcl_visualizers[index]->removePolygonMesh("meshes");
        if (!pcl_visualizers[index]->updatePolygonMesh(polygon_meshes[index],
                                                       "meshes")) {
          pcl_visualizers[index]->addPolygonMesh(polygon_meshes[index],
                                                 "meshes", 0);
        }
        
        if (save_visualizer_frames_) {
          pcl_visualizers[index]->saveScreenshot(
              "vpp_map_" + std::to_string(index) + "/frame_" +
              std::to_string(frame_count_) + ".png");
        }
      }

      frame_count_++;

      refresh = false;
    }
  }
}
}  // namespace voxblox
