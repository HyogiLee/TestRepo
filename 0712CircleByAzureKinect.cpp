// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <string>
#include <numbers>
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <k4a/k4atypes.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <thread>
#include <opencv2/opencv.hpp>
#include "open3d/Open3D.h"
#include <fstream>
//#include <conio.h>
#include <iostream>
#include <algorithm>
using namespace std;
using namespace open3d;
using namespace open3d::geometry;
using namespace open3d::visualization;

#define WIDTH 1280
#define HEIGHT 960
#define PI 3.141592653589793

/*실제 원의 크기*/
enum CIRCLETYPE {
    SMALLEST_CIRCLE = 25,
    SMALL_CIRCLE = 40,
    BIG_CIRCLE = 100,
    BIGGEST_CIRCLE = 290
};

enum CIRCLENUMBER {
    CircleCenter0 = 0,
    CircleCenter1 = 1,
    CircleCenter2 = 2,
    CircleCenter3 = 3,
    CircleCenter4 = 4,
    CircleCenter5 = 5,
    CircleCenter6 = 6,
    CircleCenter7 = 7,
    CircleCenter8 = 8
};



struct CirclesCenter {
    Eigen::Vector3d CircleCenter0;
    Eigen::Vector3d CircleCenter1;
    Eigen::Vector3d CircleCenter2;
    Eigen::Vector3d CircleCenter3;
    Eigen::Vector3d CircleCenter4;
    Eigen::Vector3d CircleCenter5;
    Eigen::Vector3d CircleCenter6;
    Eigen::Vector3d CircleCenter7;
    Eigen::Vector3d CircleCenter8;

    //표준편차 작으면 업데이트하겠다 뜻으로.
    double devi0 = 100;
    double devi1 = 100;
    double devi2 = 100;
    double devi3 = 100;
    double devi4 = 100;
    double devi5 = 100;
    double devi6 = 100;
    double devi7 = 100;
    double devi8 = 100;
    
    //거리의 합이 작으면 업데이트 하겠다.
    // 들어가는 값이 int 여서 에러났었음.
    double sum1 = 100000.0;
    double sum2 = 100000.0;
    double sum3 = 100000.0;
    double sum4 = 100000.0;
    double sum5 = 100000.0;
    double sum6 = 100000.0;
    double sum7 = 100000.0;
    double sum8 = 100000.0;
};
CirclesCenter BestCenter;

const std::string CLOUD_NAME = "points";

int max_depth = 600;
Eigen::Matrix3d matrix;

void on_max_depth_change(int pos, void* data)
{
    max_depth = pos;
}

void generate_point_cloud(geometry::PointCloud* pointData,
    const k4a_image_t depth_image,
    const k4a_image_t color_image) {
    int nCntDet = 0;
    bool status = false;
    int width = k4a_image_get_width_pixels(depth_image);
    int height = k4a_image_get_height_pixels(depth_image);
    int size = width * height;
    int16_t* depth_data = (int16_t*)(void*)k4a_image_get_buffer(depth_image);
    uint8_t* color_data = k4a_image_get_buffer(color_image);
    int count = 0;
    for (int i = 0; i < size; i++)
    {
        if (depth_data[3 * i + 2] == 0) continue;
        Eigen::Vector3d point;
        point.x() = depth_data[3 * i + 0];
        point.y() = depth_data[3 * i + 1];
        point.z() = depth_data[3 * i + 2];
        pointData->points_.push_back(point);

        Eigen::Vector3d color(color_data[4 * i + 2] / 255.0,
            color_data[4 * i + 1] / 255.0,
            color_data[4 * i + 0] / 255.0);

        pointData->colors_.push_back(color);
        count++;
    }
    std::cout << "size-" << count << endl;
}

void trim_image_data(const k4a_image_t depth_image, int16_t* depth_data, uint8_t* color_data, int size)
{
    // if depth value is bigger than 0.5m, then remove the rgbd information. if not, keeping the data
    
    int sizeofdepth = sizeof(depth_data);
    int16_t* position_values = (int16_t*)(void*)k4a_image_get_buffer(depth_image);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++)
    {
        if (position_values[3*i+2] > max_depth)
        {

            depth_data[i] = 0;
            
            position_values[3 * i + 0] = 0;
            position_values[3 * i + 1] = 0;
            position_values[3 * i + 2] = 0;

            color_data[(4 * i) + 2] = 0;
            color_data[(4 * i) + 1] = 0;
            color_data[(4 * i) + 0] = 0;
        }
        else
        {
            continue;
        }
    }
}



double gaussianRandom(double average, double stdev) {
    double v1, v2, s, temp;
    do {
        v1 = 2 * ((double)rand() / RAND_MAX) - 1;  // -1.0 ~ 1.0 까지의 값
        v2 = 2 * ((double)rand() / RAND_MAX) - 1;  // -1.0 ~ 1.0 까지의 값
        s = v1 * v1 + v2 * v2;
    } while (s >= 1 || s == 0);
    s = sqrt((-2 * log(s)) / s);
    temp = v1 * s;
    temp = (stdev * temp) + average;

    return temp;
    /*참고사이트*/
    //http://mwultong.blogspot.com/2006/11/c-vc-gaussian-random-number.html
    //.
}

 tuple<double, double> GetRadius(Eigen::Vector3d center, std::shared_ptr<PointCloud> pcd_ptr, int CIRCLE_TYPE) {
    double radius = CIRCLE_TYPE / 2 * 1.1 ;
    double sum=0;
    auto sphere = geometry::TriangleMesh::CreateSphere(radius);

    sphere->Translate(center);
    AxisAlignedBoundingBox box = sphere->GetAxisAlignedBoundingBox();
    auto a = pcd_ptr->Crop(box);
    //visualization::DrawGeometries({a});
    for (auto pt : a->points_) {
        double dist = sqrt(pow((center.x() - pt.x()), 2) +
                           pow((center.y() - pt.y()), 2) +
                           pow((center.z() - pt.z()), 2));
        sum += dist;
        if (dist < radius) radius = dist;
    }
    sum = sum / a->points_.size();
    return make_tuple(radius,sum);
}

void FindCirclesCenter(std::shared_ptr<geometry::PointCloud> source, geometry::PointCloud CirclesCenter) {
    Eigen::Matrix3d ZRotaion_matrix;
    int theta;
    bool odd = true;    //CirclesCenter pcd 에는 홀수에 SMALL CIRCLE , 짝수에 SMALLEST CIRCLE이 들어가있다. 
    double r;
    double devi;
    double sum=0;
    double avgdev = 0;
    double BeforeAvgdev=100;
    int BestTheta=-1;
    tuple<double, double> ResultOfGetRadius;
    std::shared_ptr<geometry::PointCloud> RotatedCirclesCenter = std::make_shared<PointCloud>();
    geometry::PointCloud OriginalCirclesCenter = CirclesCenter;
    geometry::PointCloud CirclesCenterChangedZ = CirclesCenter;
    *RotatedCirclesCenter = CirclesCenterChangedZ;
    //double z = CirclesCenterChangedZ.points_.at(0).z(); //center의 z 값 상에 위치해야함. z 축 기준으로 회전시키면서 할꺼니까. 
    CIRCLENUMBER circlenumber;

    /*0712 z 축의 두께 생각 처음에 1mm 라고 설정을 했으므로 얇은 disk 생각 그러면 그 두께만큼 z 값을 생각해주어야한다.*/
    /*xy z축으로 세워도 완전한 z축이 아님, 따라서 xy 평면에서 회전시키는 것으로는 부족함. 오차를 줄이기위해서는 그 부근의 점들에 대해서 가장 최선의 결과를 나타내는 점을 찾으면됨.*/
    //method 1. Circle Center의 z 값의 범위 설정
    //method 2. yz, xz 평면에서의 회전
    //method 3. 처음 Circle Center 에서, 그 부근의 범위안에 있는 포인트들에 대해서 조사.
        
    for (double z = - 1; z <  1; z = z + 0.1) {
        *RotatedCirclesCenter = RotatedCirclesCenter->Rotate(matrix, RotatedCirclesCenter->points_.at(0));
        for (int CenterNum = 0; CenterNum < 9; CenterNum++) {
            Eigen::Vector3d TempCircleCenter =
            RotatedCirclesCenter->points_.at(CenterNum);
            TempCircleCenter(2) += z;
            RotatedCirclesCenter->points_.at(CenterNum) = TempCircleCenter;
            //*RotatedCirclesCenter = RotatedCirclesCenter->Translate(RotatedCirclesCenter->points_.at(0));
        }
        *RotatedCirclesCenter = RotatedCirclesCenter->Rotate(matrix.inverse(), RotatedCirclesCenter->points_.at(0));

        for (theta = -15; theta < 15; theta++) {
            ZRotaion_matrix << cos(theta * PI / 180),
                    (-1) * sin(theta * PI / 180), 0, sin(theta * PI / 180),
                    cos(theta * PI / 180), 0, 0, 0, 1;
            *RotatedCirclesCenter = RotatedCirclesCenter->Rotate(ZRotaion_matrix, RotatedCirclesCenter->points_.at(0));

            for (int CenterNum = 0; CenterNum < 9; CenterNum++) {
                Eigen::Vector3d CircleCenter =
                        RotatedCirclesCenter->points_.at(CenterNum);

                switch (CenterNum) {
                    case CIRCLENUMBER::CircleCenter0:
                        ResultOfGetRadius = GetRadius(CircleCenter, source,
                                                      CIRCLETYPE::BIG_CIRCLE);
                        r = get<0>(ResultOfGetRadius);
                        devi = abs(r - (CIRCLETYPE::BIG_CIRCLE / 2));
                        if (BestCenter.devi0 > devi) {
                            BestCenter.devi0 = devi;
                            BestCenter.CircleCenter0 = CircleCenter;
                        }
                        break;
                    case CIRCLENUMBER::CircleCenter1:
                        ResultOfGetRadius = GetRadius(CircleCenter, source,
                                                      CIRCLETYPE::SMALL_CIRCLE);
                        r = get<0>(ResultOfGetRadius);
                        sum = get<1>(ResultOfGetRadius);
                        devi = abs(r - (CIRCLETYPE::SMALL_CIRCLE / 2));
                        // cout << "Circle 1  theta : " << theta << " r: " << r
                        //     << "  sum : " << sum << endl;
                        if (BestCenter.devi1 > devi) {
                            BestCenter.devi1 = devi;
                            BestCenter.sum1 = sum;
                            BestCenter.CircleCenter1 = CircleCenter;
                        }
                        break;
                    case CIRCLENUMBER::CircleCenter2:
                        ResultOfGetRadius =
                                GetRadius(CircleCenter, source,
                                          CIRCLETYPE::SMALLEST_CIRCLE);
                        r = get<0>(ResultOfGetRadius);
                        sum = get<1>(ResultOfGetRadius);
                        devi = abs(r - (CIRCLETYPE::SMALLEST_CIRCLE / 2));
                        // cout << "Circle 2 theta : " << theta << " r: " << r
                        //     << "  sum : " << sum << endl;
                        if (BestCenter.devi2 > devi) {
                            BestCenter.devi2 = devi;
                            BestCenter.sum2 = sum;
                            BestCenter.CircleCenter2 = CircleCenter;
                        }
                        break;
                    case CIRCLENUMBER::CircleCenter3:
                        ResultOfGetRadius = GetRadius(CircleCenter, source,
                                                      CIRCLETYPE::SMALL_CIRCLE);
                        r = get<0>(ResultOfGetRadius);
                        sum = get<1>(ResultOfGetRadius);
                        devi = abs(r - (CIRCLETYPE::SMALL_CIRCLE / 2));
                        if (BestCenter.devi3 > devi) {
                            BestCenter.devi3 = devi;
                            BestCenter.sum3 = sum;
                            BestCenter.CircleCenter3 = CircleCenter;
                        }
                        break;
                    case CIRCLENUMBER::CircleCenter4:
                        ResultOfGetRadius =
                                GetRadius(CircleCenter, source,
                                          CIRCLETYPE::SMALLEST_CIRCLE);
                        r = get<0>(ResultOfGetRadius);
                        sum = get<1>(ResultOfGetRadius);
                        devi = abs(r - (CIRCLETYPE::SMALLEST_CIRCLE / 2));
                        if (BestCenter.devi4 > devi) {
                            BestCenter.devi4 = devi;
                            BestCenter.sum4 = sum;
                            BestCenter.CircleCenter4 = CircleCenter;
                        }
                        break;
                    case CIRCLENUMBER::CircleCenter5:
                        ResultOfGetRadius = GetRadius(CircleCenter, source,
                                                      CIRCLETYPE::SMALL_CIRCLE);
                        r = get<0>(ResultOfGetRadius);
                        sum = get<1>(ResultOfGetRadius);
                        devi = abs(r - (CIRCLETYPE::SMALL_CIRCLE / 2));
                        if (BestCenter.devi5 > devi) {
                            BestCenter.devi5 = devi;
                            BestCenter.sum5 = sum;
                            BestCenter.CircleCenter5 = CircleCenter;
                        }
                        break;
                    case CIRCLENUMBER::CircleCenter6:
                        ResultOfGetRadius =
                                GetRadius(CircleCenter, source,
                                          CIRCLETYPE::SMALLEST_CIRCLE);
                        r = get<0>(ResultOfGetRadius);
                        sum = get<1>(ResultOfGetRadius);
                        devi = abs(r - (CIRCLETYPE::SMALLEST_CIRCLE / 2));
                        if (BestCenter.devi6 > devi) {
                            BestCenter.devi6 = devi;
                            BestCenter.sum6 = sum;
                            BestCenter.CircleCenter6 = CircleCenter;
                        }
                        break;
                    case CIRCLENUMBER::CircleCenter7:
                        ResultOfGetRadius = GetRadius(CircleCenter, source,
                                                      CIRCLETYPE::SMALL_CIRCLE);
                        r = get<0>(ResultOfGetRadius);
                        sum = get<1>(ResultOfGetRadius);
                        devi = abs(r - (CIRCLETYPE::SMALL_CIRCLE / 2));
                        if (BestCenter.devi7 > devi) {
                            BestCenter.devi7 = devi;
                            BestCenter.sum7 = sum;
                            BestCenter.CircleCenter7 = CircleCenter;
                        }
                        break;
                    case CIRCLENUMBER::CircleCenter8:
                        ResultOfGetRadius =
                                GetRadius(CircleCenter, source,
                                          CIRCLETYPE::SMALLEST_CIRCLE);
                        r = get<0>(ResultOfGetRadius);
                        sum = get<1>(ResultOfGetRadius);
                        devi = abs(r - (CIRCLETYPE::SMALLEST_CIRCLE / 2));
                        if (BestCenter.devi8 > devi) {
                            BestCenter.devi8 = devi;
                            BestCenter.sum8 = sum;
                            BestCenter.CircleCenter8 = CircleCenter;
                        }
                        break;
                }
            }
        }
    }

    
    



    //여기서 이뤄지려면 어떻게?? clear 가 안되고 쌓이기만함...
    CirclesCenter.Clear();
    CirclesCenter.points_.push_back(BestCenter.CircleCenter0);
    CirclesCenter.points_.push_back(BestCenter.CircleCenter1);
    CirclesCenter.points_.push_back(BestCenter.CircleCenter2);
    CirclesCenter.points_.push_back(BestCenter.CircleCenter3);
    CirclesCenter.points_.push_back(BestCenter.CircleCenter4);
    CirclesCenter.points_.push_back(BestCenter.CircleCenter5);
    CirclesCenter.points_.push_back(BestCenter.CircleCenter6);
    CirclesCenter.points_.push_back(BestCenter.CircleCenter7);
    CirclesCenter.points_.push_back(BestCenter.CircleCenter8);
    for (int i = 0; i < 9; i++) {
        CirclesCenter.colors_.push_back({1, 0, 0});
    }
    //ZRotaion_matrix << cos(BestTheta * PI / 180), (-1) * sin(BestTheta * PI / 180), 0,
    //                   sin(BestTheta * PI / 180), cos(BestTheta * PI / 180), 0,
    //                   0, 0, 1;
    //*CirclesCenter = CirclesCenter->Rotate(ZRotaion_matrix,CirclesCenter->points_.at(0));

    
    /*Current Deviation Check*/
    std::cout << "\nCenter Number : " << 0 << "  표준편차 : " << BestCenter.devi0 << endl;
    std::cout << "Center Number : " << 1 << "  표준편차 : " << BestCenter.devi1 << endl;
    std::cout << "Center Number : " << 2 << "  표준편차 : " << BestCenter.devi2 << endl;
    std::cout << "Center Number : " << 3 << "  표준편차 : " << BestCenter.devi3 << endl;
    std::cout << "Center Number : " << 4 << "  표준편차 : " << BestCenter.devi4<< endl;
    std::cout << "Center Number : " << 5 << "  표준편차 : " << BestCenter.devi5<< endl;
    std::cout << "Center Number : " << 6 << "  표준편차 : " << BestCenter.devi6 << endl;
    std::cout << "Center Number : " << 7 << "  표준편차 : " << BestCenter.devi7 << endl;
    std::cout << "Center Number : " << 8 << "  표준편차 : " << BestCenter.devi8 << endl;

    //cout << "\nCenter Number : " << 0 << "  표준편차 : " << BestCenter.devi0 << endl;
    //cout << "Center Number : " << 1 << "  표준편차 : " << BestCenter.devi1
    //     << " Sum of Distatnces : " << BestCenter.sum1 << endl;
    //cout << "Center Number : " << 2 << "  표준편차 : " << BestCenter.devi2
    //     << " Sum of Distatnces : " << BestCenter.sum2 << endl;
    //cout << "Center Number : " << 3 << "  표준편차 : " << BestCenter.devi3
    //     << " Sum of Distatnces : " << BestCenter.sum3 << endl;
    //cout << "Center Number : " << 4 << "  표준편차 : " << BestCenter.devi4
    //     << " Sum of Distatnces : " << BestCenter.sum4 << endl;
    //cout << "Center Number : " << 5 << "  표준편차 : " << BestCenter.devi5
    //     << " Sum of Distatnces : " << BestCenter.sum5 << endl;
    //cout << "Center Number : " << 6 << "  표준편차 : " << BestCenter.devi6
    //     << " Sum of Distatnces : " << BestCenter.sum6 << endl;
    //cout << "Center Number : " << 7 << "  표준편차 : " << BestCenter.devi7
    //     << " Sum of Distatnces : " << BestCenter.sum7 << endl;
    //cout << "Center Number : " << 8 << "  표준편차 : " << BestCenter.devi8
    //     << " Sum of Distatnces : " << BestCenter.sum8 << endl;
       
    //for (auto CircleCenter : CirclesCenter->points_) {
    //    if (CircleCenter == RotatedCirclesCenter->points_.at(0)) {
    //        r = GetRadius(CircleCenter, source, CIRCLETYPE::BIG_CIRCLE);
    //        cout << "\n Radius : " << r << "  표준편차 : " << abs(r - (CIRCLETYPE::BIG_CIRCLE / 2)) << endl;
    //        continue;
    //    }
    //    if (odd) {
    //        r = GetRadius(CircleCenter, source, CIRCLETYPE::SMALL_CIRCLE);
    //        cout << " Radius : " << r << "  표준편차 : " << abs(r - (CIRCLETYPE::SMALL_CIRCLE / 2)) << endl;
    //        odd = false;
    //    } else {
    //        r = GetRadius(CircleCenter, source, CIRCLETYPE::SMALLEST_CIRCLE);            
    //        cout << " Radius : " << r << "  표준편차 : " << abs(r - (CIRCLETYPE::SMALLEST_CIRCLE / 2)) << endl;
    //        odd = true;
    //    }
    //}
    //visualization::DrawGeometries({source});
}



#if 1
int main(int argc, const char* argv[]) {
    geometry::PointCloud pointData;
    uint32_t device_count = k4a_device_get_installed_count();
    k4a_device_t device = NULL;
    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    config.color_format = k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_BGRA32;
    config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
    config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;  // No need for depth during calibration
    config.camera_fps = K4A_FRAMES_PER_SECOND_5;  // Don't use all USB bandwidth
    config.subordinate_delay_off_master_usec = 0;  // Must be zero for master
    config.synchronized_images_only = true;  // ensures that depth and color images are both available in the capture

    cv::namedWindow("image1");

    cv::createTrackbar("max depth", "image1", &max_depth, 1000, on_max_depth_change, nullptr);

    if (device_count == 0) {
        std::cout << "No K4A devices found" << endl;
        return 1;
    }
    if (K4A_RESULT_SUCCEEDED != k4a_device_open(K4A_DEVICE_DEFAULT, &device)) {
        std::cout << "Failed to open device" << endl;
        k4a_device_close(device);
        return 1;
    }


    if (device != NULL) std::cout << "device open success" << endl;

    // Retrive calibration
    k4a_calibration_t calibration;
    if (K4A_RESULT_SUCCEEDED !=
        k4a_device_get_calibration(device, config.depth_mode,
                                   config.color_resolution, &calibration)) {
        cout << "Failed to get calibration" << endl;
        k4a_device_close(device);
        return 1;
    }
    cout << "calibration success" << endl;
    cout << calibration.color_camera_calibration.intrinsics.parameters.param.fx << endl;

    unsigned char json[10240] = { 0, };
    //std::unique_ptr<unsigned char> json = std::make_unique<unsigned char>(1024);

    size_t size_json = sizeof(json);
    k4a_buffer_result_t result_buffer = k4a_device_get_raw_calibration(device, json, &size_json);
    
    ofstream writeFile("calibration.json");
    if (writeFile.is_open()) {
        writeFile << json << endl;
        writeFile.close();
    }

    char sn[20] = { 0, };
    size_t size = 0;

    size_t serial_number_length = 0;

    if (K4A_BUFFER_RESULT_TOO_SMALL !=
        k4a_device_get_serialnum(device, NULL, &serial_number_length)) {
        k4a_device_close(device);
        return 1;
    }

    char* serial_number = new (std::nothrow) char[serial_number_length];

    if (serial_number == NULL) {
        k4a_device_close(device);
    }

    if (K4A_BUFFER_RESULT_SUCCEEDED !=
        k4a_device_get_serialnum(device, serial_number,
                                 &serial_number_length)) {
        delete[] serial_number;
        serial_number = NULL;
        k4a_device_close(device);
    }

    if (serial_number[0] != 0)
        cout << serial_number << " detected" << endl;
    else
        cout << "camera not detected" << endl;

    if (K4A_RESULT_SUCCEEDED != k4a_device_start_cameras(device, &config))
    {
        cout << "Failed to start device" << endl;
        k4a_device_close(device);
        return 1;
    }

    k4a_transformation_t transformation;
    transformation = k4a_transformation_create(&calibration);

    cout << "transformation success" << endl;

    k4a_capture_t capture = NULL;
    k4a_image_t depth_image = NULL;
    k4a_image_t transformed_depth_image = NULL;
    k4a_image_t color_image = NULL;
    const int32_t TIMEOUT_IN_MS = 5000;
    bool isRunning = true;
    bool isGeomatryAdded = false;


    visualization::VisualizerWithKeyCallback vis{};
    auto callback_exit = [&](visualization::Visualizer* vis)
    {
        isRunning = false;
        if (isRunning) {
            cout << "Loop will be started" << endl;
        }
        else {
            cout << "Loop will be ended" << endl;
        }
        return false;
    };

    int i = 0;
    std::shared_ptr<geometry::PointCloud> pointcloud_ptr(new geometry::PointCloud);
    geometry::PointCloud pointData_original;
    std::shared_ptr<PointCloud> inlier_cloud(new geometry::PointCloud);
    std::shared_ptr<PointCloud> transformed_inlier_cloud(new geometry::PointCloud);
    std::shared_ptr<geometry::PointCloud> temp_ptr = std::make_shared<geometry::PointCloud>();
    std::shared_ptr<geometry::PointCloud> CirclesCenter_ptr = std::make_shared<geometry::PointCloud>();
    std::shared_ptr<geometry::PointCloud> pcd_ptr = std::make_shared<geometry::PointCloud>();
    std::shared_ptr<vector<geometry::PointCloud>> clustered_pcd = std::make_shared<vector<geometry::PointCloud>>() ;
    std::shared_ptr<geometry::AxisAlignedBoundingBox> box_ptr = std::make_shared<geometry::AxisAlignedBoundingBox>();
    std::shared_ptr<geometry::TriangleMesh> center_ptr =std::make_shared<geometry::TriangleMesh>();
    Eigen::Vector3d center;

    auto callback_pcd_save = [&](visualization::Visualizer* vis) {
        std::string filepath = "d:\\";
        std::time_t currTime = std::time(nullptr);
        tm* ltm = localtime(&currTime);
        char* dt = ctime(&currTime);
        string filename(dt);
        filename.erase(std::remove(filename.begin(), filename.end(), '\n'), filename.end());
        filename.erase(std::remove(filename.begin(), filename.end(), ':'), filename.end());

        filename = filename + ".pcd";
        i++;
        cout << filepath + filename << endl;
        io::WritePointCloud(filepath + filename, *inlier_cloud);
        return true;
    };

    vis.RegisterKeyCallback(GLFW_KEY_ESCAPE, callback_exit);
    vis.RegisterKeyCallback(GLFW_KEY_S, callback_pcd_save);
    
    int nCount = 0;

    /*원의 중심점 간의 거리 이용해서 center 에서부터 떨어진위치 설정*/
    double length = 102.0;

    while (isRunning) {

        switch (k4a_device_get_capture(device, &capture, TIMEOUT_IN_MS)) {
        case K4A_WAIT_RESULT_SUCCEEDED:
            break;
        case K4A_WAIT_RESULT_TIMEOUT:
            //printf("Timed out waiting for a capture\n");
            continue;
            break;
        case K4A_WAIT_RESULT_FAILED:
            //printf("Failed to read a capture\n");
            k4a_device_close(device);
            k4a_capture_release(capture);
            return 1;
        }

        // Retrieve depth image
        try {
            depth_image = k4a_capture_get_depth_image(capture);
            if (depth_image == NULL) {
                cout << "Depth16 None" << endl;
                k4a_capture_release(capture);
                continue;
            }
            //cout << "get depth_image success" << endl;

        }
        catch (...) {
            k4a_capture_release(capture);
            continue;
        }

        try {
            color_image = k4a_capture_get_color_image(capture);
            if (color_image == NULL) {
                cout << "Color None" << endl;
                k4a_capture_release(capture);
                continue;
            }
            //cout << "get color_image success" << endl;

        }
        catch (...) {
            k4a_capture_release(capture);
            continue;
        }

        int width = k4a_image_get_width_pixels(color_image);
        int height = k4a_image_get_height_pixels(color_image);


        if (width == 0 || height == 0)
            continue;

        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, width, height, width * (int)sizeof(uint16_t), &transformed_depth_image))
        {
            k4a_image_release(transformed_depth_image);
            k4a_image_release(color_image);
            k4a_image_release(depth_image);
            k4a_capture_release(capture);
            continue;
        }

        if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_color_camera(transformation, depth_image, transformed_depth_image))
        {
            k4a_image_release(transformed_depth_image);
            k4a_image_release(color_image);
            k4a_image_release(depth_image);
            k4a_capture_release(capture);
            continue;
        }

        k4a_image_t point_cloud_image = nullptr;

        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM, width, height, width * 3 * (int)sizeof(int16_t), &point_cloud_image))
        {
            k4a_image_release(point_cloud_image);
            k4a_image_release(depth_image);
            k4a_image_release(color_image);
            k4a_capture_release(capture);
            continue;
        }

        if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_point_cloud(transformation, transformed_depth_image, K4A_CALIBRATION_TYPE_COLOR, point_cloud_image))
        {
            k4a_image_release(point_cloud_image);
            k4a_image_release(color_image);
            k4a_image_release(depth_image);
            k4a_capture_release(capture);
            continue;
        }

        width = k4a_image_get_width_pixels(transformed_depth_image);
        height = k4a_image_get_height_pixels(transformed_depth_image);

        int16_t* depth_data = (int16_t*)(void*)k4a_image_get_buffer(transformed_depth_image);
        uint8_t* color_data = k4a_image_get_buffer(color_image);
        trim_image_data(point_cloud_image, depth_data, color_data, width * height);
        generate_point_cloud(&pointData, point_cloud_image, color_image);
        pointData_original = pointData;
        *pointcloud_ptr = pointData_original;
        //width = k4a_image_get_width_pixels(depth_image);
        //height = k4a_image_get_height_pixels(depth_image);


        cv::Mat depthMap(height, width, CV_16U, (void*)k4a_image_get_buffer(transformed_depth_image), cv::Mat::AUTO_STEP);
        //cv::Mat depthMap(height, width, CV_16U, (void*)k4a_image_get_buffer(depth_image), cv::Mat::AUTO_STEP);
        cv::Mat colorMap(height, width, CV_8UC4, (void*)k4a_image_get_buffer(color_image), cv::Mat::AUTO_STEP);

        int max_rows = colorMap.rows;
        int max_cols = colorMap.cols;
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < max_rows; i++)
        {
            for(int j=0; j<max_cols; j++)
            {
                if(depthMap.at<uint16_t>(i,j)==0)
                {
                    cv::Vec4b& color = colorMap.at<cv::Vec4b>(i,j);
                    color = { 0,0,0,0 };
                }
            }
        }
        uint64_t tmstamp = k4a_image_get_timestamp_usec(color_image);
        std::string strTmStamp = std::to_string((unsigned long long)tmstamp);
        
        std::string filePath="d:\\TestData\\AzureKinect\\Sphere\\";
        std::stringstream depthfileName;
        std::stringstream colorfileName;
        //depthfileName << filePath+"Depth\\" << std::setw(4) << std::setfill('0') << nCount   << ".png";
        //colorfileName << filePath + "RGB\\" << std::setw(4) << std::setfill('0') << nCount++ << ".png";

        depthfileName << filePath+"Depth\\" << strTmStamp << ".png";
        colorfileName << filePath + "RGB\\" << strTmStamp << ".png";

        cv::imshow("Color", colorMap);
        cv::imwrite(colorfileName.str().c_str(), colorMap);

/*여기서 matrix를 바꿔주기위해서 아래 코드를 사용한다.*/
        
        *pcd_ptr = pointData;
        std::tuple<Eigen::Vector4d, std::vector<size_t>> plane_model = pcd_ptr->SegmentPlane(1.0);
        pcd_ptr = pcd_ptr->SelectByIndex(std::get<1>(plane_model));
        //visualization::DrawGeometries({pcd_ptr});

        // Data Clustering
        double eps = 10;
        size_t min_points = 20;
        std::vector<int> indice = pcd_ptr->ClusterDBSCAN(eps, min_points);
        int max_num = *std::max_element(indice.begin(), indice.end());
        if(max_num < 0) {
            std::cerr << "error to find clustring data" << std::endl;
            return -1;
        }
        int point_no_size = indice.size();
        for (int i = 0; i <= max_num; i++) {
            PointCloud data = PointCloud();
            for (int j = 0; j < point_no_size; j++) {
                if (indice.at(j) == i) {
                    data.points_.push_back(pcd_ptr->points_.at(j));
                }
            }
            clustered_pcd->push_back(data);
        }


        // Find the largest dataset
        int max_index=-1;
        int max_point_num = 0;

        for (int i = 0; i <= max_num; i++) {
            if (clustered_pcd->at(i).points_.size() > max_point_num) {
                max_point_num = clustered_pcd->at(i).points_.size();
                max_index = i;
            }
        }
        if (max_index < 0) {
            std::cerr << "error to find the largest dataset" << std::endl;
            return -1;
        }
        *inlier_cloud = clustered_pcd->at(max_index);

        //temp_ptr->PaintUniformColor({0, 0, 1});
        Eigen::Vector4d equation = std::get<0>(plane_model);

        auto box = inlier_cloud->GetAxisAlignedBoundingBox();
        box_ptr->min_bound_ = box.min_bound_;
        box_ptr->max_bound_ = box.max_bound_;
        box_ptr->color_ = {1, 0, 0};
        center = (box.min_bound_ + box.max_bound_) / 2;

        open3d::geometry::KDTreeFlann pcd_tree(*inlier_cloud);

        std::vector<int> nbs;
        std::vector<double> dists2;
        pcd_tree.SearchRadius(center, 55, nbs, dists2);

        if(nbs.size() <= 0) {
            std::cerr << "Can't find the nearest points" << std::endl;
            return -1;
        }

        PointCloud icp_src;
        Eigen::Vector3d src_origin_p(center);
        Eigen::Vector3d src_z_axis_p = src_origin_p + Eigen::Vector3d(equation.x(), equation.y(), equation.z()).normalized(); 
        Eigen::Vector3d src_y_axis_p =src_origin_p + (inlier_cloud->points_.at(nbs[0]) -src_origin_p).normalized(); 
        Eigen::Vector3d src_x_axis_p =src_origin_p + src_z_axis_p.cross(src_y_axis_p).normalized();
        icp_src.points_ = {src_origin_p, src_x_axis_p, src_y_axis_p, src_z_axis_p};

        PointCloud icp_dst;
        Eigen::Vector3d dst_origin_p(center);
        Eigen::Vector3d dst_z_axis_p(dst_origin_p + Eigen::Vector3d(0, 0, 1));
        Eigen::Vector3d dst_y_axis_p(dst_origin_p + Eigen::Vector3d(0, 1, 0));
        Eigen::Vector3d dst_x_axis_p(dst_origin_p + Eigen::Vector3d(1, 0, 0));
        icp_dst.points_ = {dst_origin_p, dst_x_axis_p, dst_y_axis_p, dst_z_axis_p};

        open3d::pipelines::registration::RegistrationResult icp_result =
        open3d::pipelines::registration::RegistrationICP( icp_src, icp_dst, 2, Eigen::Matrix4d::Identity(),
        open3d::pipelines::registration::TransformationEstimationPointToPoint(),
        open3d::pipelines::registration::ICPConvergenceCriteria());
        //cout << icp_result.fitness_ << endl;
        //cout << icp_result.inlier_rmse_ << endl;
        Eigen::Matrix4d_u transform = icp_result.transformation_;
        Eigen::Matrix4d_u invtransform = transform.inverse();
        *transformed_inlier_cloud = inlier_cloud->Transform(transform);

        /*법선 벡터 확인.*/
        //auto mesh = geometry::TriangleMesh::CreateCoordinateFrame(500);
        //std::tuple<Eigen::Vector4d, std::vector<size_t>> plane_model1 = inlier_cloud->SegmentPlane(0.5);
        //Eigen::Vector4d equation1 = std::get<0>(plane_model1);
        //cout << "original equation  : " << equation1 << endl;
        //visualization::DrawGeometries({inlier_cloud, mesh});
        //inlier_cloud->Transform(transform);
        //std::tuple<Eigen::Vector4d, std::vector<size_t>> plane_model2 = inlier_cloud->SegmentPlane(0.5);
        //Eigen::Vector4d equation2 = std::get<0>(plane_model2);
        //cout << "transformed equation  : " << equation2 << endl;
        //visualization::DrawGeometries({inlier_cloud, mesh});
        //inlier_cloud->Transform(invtransform);
        //std::tuple<Eigen::Vector4d, std::vector<size_t>> plane_model3 = inlier_cloud->SegmentPlane(0.5);
        //Eigen::Vector4d equation3 = std::get<0>(plane_model3);
        //cout << "inverse transformed equation  : " << equation3 << endl;
        //visualization::DrawGeometries({inlier_cloud, mesh});

        // CirclesCenter_ptr에 원들의 중심점(9개) 다 넣어줌.
        CirclesCenter_ptr->Clear();
        for (int i = 0; i < 9; i++) {
            Eigen::Vector3d CircleCenter;                           
            if (i == 0) {
                CirclesCenter_ptr->points_.push_back({0,0,0});        
                continue;
            }
            int j = i - 1;
            double x_length = length * cos(PI / 4 * j);
            double y_length = length * sin(PI / 4 * j);
            CircleCenter.x() =  x_length ;
            CircleCenter.y() =  y_length  ;
            CircleCenter.z() =  0 ; 
            CirclesCenter_ptr->points_.push_back(CircleCenter);
        }

        std::tuple<Eigen::Vector4d, std::vector<size_t>> plane_model2 = transformed_inlier_cloud->SegmentPlane(0.5);
        Eigen::Vector4d equation2 = std::get<0>(plane_model2);
        matrix << 1, 0, equation2.x() / equation2.w(), 0, 1,
                equation2.y() / equation2.w(), 0, 0, equation2.z() / equation2.w();

        *CirclesCenter_ptr = CirclesCenter_ptr->Rotate(matrix.inverse(), {0,0,0});
        *CirclesCenter_ptr = CirclesCenter_ptr->Translate(center);
        CirclesCenter_ptr->PaintUniformColor({1, 0,0});
        //CirclesCenter_ptr->Transform(transform);
        //auto frame = geometry::TriangleMesh::CreateCoordinateFrame(300);
        //*transformed_inlier_cloud = inlier_cloud->Transform(invtransform);

       //open3d::visualization::DrawGeometries({frame,transformed_inlier_cloud,CirclesCenter_ptr});

        //z축으로 기울어진 pcd에 원들의 중심점 CirclesCenter_ptr을 넣고 각도를 돌려보면서 가장 잘맞는 center를 찾고 업데이트해줌.
        FindCirclesCenter(transformed_inlier_cloud, *CirclesCenter_ptr);     

        /*결과 확인*/
        CirclesCenter_ptr->Clear();
        CirclesCenter_ptr->points_.push_back(BestCenter.CircleCenter0);
        CirclesCenter_ptr->points_.push_back(BestCenter.CircleCenter1);
        CirclesCenter_ptr->points_.push_back(BestCenter.CircleCenter2);
        CirclesCenter_ptr->points_.push_back(BestCenter.CircleCenter3);
        CirclesCenter_ptr->points_.push_back(BestCenter.CircleCenter4);
        CirclesCenter_ptr->points_.push_back(BestCenter.CircleCenter5);
        CirclesCenter_ptr->points_.push_back(BestCenter.CircleCenter6);
        CirclesCenter_ptr->points_.push_back(BestCenter.CircleCenter7);
        CirclesCenter_ptr->points_.push_back(BestCenter.CircleCenter8);
        for (int i = 0; i < 9; i++) {
            CirclesCenter_ptr->colors_.push_back({1, 0, 0});
        }
        //visualization::DrawGeometries({transformed_inlier_cloud,CirclesCenter_ptr});

        io::WritePointCloudToPCD("D:\\original.pcd", *pointcloud_ptr, true);
        io::WritePointCloudToPCD("D:\\transformed_inlier_cloud.pcd", *transformed_inlier_cloud, true);
        io::WritePointCloudToPCD("D:\\before_CirclesCenter.pcd", *CirclesCenter_ptr, true);

        *CirclesCenter_ptr = CirclesCenter_ptr->Transform(invtransform);
        *transformed_inlier_cloud = transformed_inlier_cloud->Transform(invtransform);
        transformed_inlier_cloud->PaintUniformColor({1, 0, 0});
        io::WritePointCloudToPCD("D:\\inverse_inlier_cloud.pcd", *transformed_inlier_cloud, true);
        io::WritePointCloudToPCD("D:\\after_CirclesCenter.pcd", *CirclesCenter_ptr, true);
        //open3d::visualization::DrawGeometries({temp_ptr, CirclesCenter_ptr});
        if (nCount == 50) break;
        nCount++;

        /*좌표계체크*/
        //auto frame = geometry::TriangleMesh::CreateCoordinateFrame(1000);
        //open3d::visualization::DrawGeometries({inlier_cloud, temp_ptr, frame});

        if (!isGeomatryAdded) {
            //vis.CreateVisualizerWindow();
            vis.CreateVisualizerWindow("Open3D with Azure Kinect", 1280, 860, 50, 50, true);
            vis.ClearGeometries();
            //vis.AddGeometry(temp_ptr);
            vis.AddGeometry(inlier_cloud);
            vis.AddGeometry(box_ptr);
            vis.AddGeometry(CirclesCenter_ptr);
            //vis.AddGeometry(CirclesCenter_ptr);
            isGeomatryAdded = true;
        }
        
        //vis.UpdateGeometry(temp_ptr);
        vis.UpdateGeometry(inlier_cloud);
        vis.UpdateGeometry(box_ptr);
        vis.UpdateGeometry(CirclesCenter_ptr);
        //vis.UpdateGeometry(CirclesCenter_ptr);
        vis.PollEvents();
        vis.UpdateRender();

        k4a_image_release(point_cloud_image);
        k4a_image_release(transformed_depth_image);
        k4a_image_release(depth_image);
        k4a_image_release(color_image);
        k4a_capture_release(capture);
        pointData.Clear();
        center_ptr->Clear();

    }
    vis.DestroyVisualizerWindow();

    //MultipleWindowsApp().Run();
    return 0;
}
#endif
