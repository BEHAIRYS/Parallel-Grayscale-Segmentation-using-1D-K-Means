//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <omp.h>
//#include <iostream>
//#include <Windows.h>
//#include <tchar.h>
//#include <random>   
//#include<opencv2/imgproc/imgproc.hpp>
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//
//   
//    cv::Mat img;
//    cv::Mat gray_img;
//    cv::imread("C:/Users/HP/Desktop/Grayscale Segmentation.jpg", cv::IMREAD_COLOR).convertTo(img, CV_32FC3, (1. / 255.));
//    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
//    //namedWindow("First OpenCV Application", WINDOW_AUTOSIZE);
//    //cv::imshow("First OpenCV Application", gray_img);
//    //cv::moveWindow("First OpenCV Application", 0, 45);
//    //cv::waitKey(0);
//    //cv::destroyAllWindows();
//    
//
//   
//    // constructing 1D array of pixel values of the image
//    float* data = new float[gray_img.rows * gray_img.cols];
//    cout << gray_img.at<float>(0,0) <<" " << gray_img.cols << endl;
//    double start_s;
//    double stop_s;
//    double TotalTime = 0.0;
//    
//
//    
//    int nthreads = 4;
//    int tid;
//    //cout << gray_img.at<float>(2,1) << endl;
//    int counter = 0;
//    start_s = clock();
//    #pragma omp parallel private(tid) shared(data,start_s,stop_s,TotalTime) num_threads(nthreads) reduction(+:counter)
//    {
//       
//        tid = omp_get_thread_num();
//        
//        int local_height = (gray_img.rows / nthreads);
//        int remainder = local_height;
//        if (tid == nthreads - 1)
//            remainder += gray_img.rows % nthreads;
//        int start = (tid * local_height);
//        int end = start + remainder;
//        
//        for (int i = start; i < end; i++)
//        {
//            for (int j = 0; j < gray_img.cols; j++) {
//                data[i*gray_img.cols +j] = gray_img.at<float>(i, j);
//            //cout << data[counter] << " ";
//            //cout << tid <<" "<<counter << endl;
//                #pragma omp atomic
//                counter++;
//            }
//        }
//       
//       
//    }
//    stop_s = clock();
//    TotalTime += (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
//
//    cout << "Time of constructing 1D array of image pixels taken by thread" << ": " << TotalTime << endl;
//         // generate random 3 centroids 
//         random_device rd; // create a random device object to seed the generator
//         mt19937 gen(rd()); // create a Mersenne Twister engine object with seed from random_device
//         uniform_int_distribution<int> distr(0, counter - 1); // create a uniform integer distribution object with range [0, counter-1]
//         cout << counter<< endl;
//         int k = 3;
//         vector<float> centroids(k);
//        
//             //store the initial centroids 
//             for (int i = 0; i < k; i++)
//             {
//                 centroids[i] = data[distr(gen)];
//                 //cout << distr(gen) << " ";
//                 cout << centroids[i]<<endl;
//             }
//         
//         int start_algo;
//         int stop_algo;
//         int TotalAlgoTime = 0;
//         
//
//    vector<int> assignments(counter);
//    bool converged = false;
//    
//    start_algo = clock();
//#pragma omp parallel private(tid) num_threads(4) shared(data,assignments,centroids)
//    while (!converged) {
//        // Assign each pixel to the closest centroid
//        tid = omp_get_thread_num();
//
//        int local_counter = (counter / nthreads);
//        int remainder = local_counter;
//        if (tid == nthreads - 1)
//            remainder += counter % nthreads;
//        int start = (tid * local_counter);
//        int end = start + remainder;
//
//        for (int i = start; i < end; i++) {
//            float min_distance = FLT_MAX;
//            int closest_centroid;
//            for (int j = 0; j < k; j++) {
//                float distance = norm(data[i] - centroids[j]);
//                if (distance < min_distance) {
//                    min_distance = distance;
//                    closest_centroid = j;
//                }
//            }
//            assignments[i] = closest_centroid;
//        }
//        vector<float> new_centroids(k);
//        vector<int> counts(k, 0);
//        
//        for (int i = start; i < end; i++) {
//            int cluster = assignments[i];
//            new_centroids[cluster] += data[i];
//            counts[cluster]++;
//        }
//        
//        for (int i = 0; i < k; i++) {
//            if (counts[i] > 0) {
//                new_centroids[i] /= counts[i];
//            }
//        }
//
//
//
//        // Check for convergence
//        float epsilon = 0.01;
//        converged = true;
//       
//        for (int i = 0; i < k; i++) {
//            if (norm(new_centroids[i] - centroids[i]) > epsilon) {
//                converged = false;
//                break;
//            }
//        }
//       
//        //update the centroids
//        #pragma omp single
//        {
//            centroids = new_centroids;
//           
//        }
//        
//        
//    }
//    stop_algo = clock();
//    TotalAlgoTime += (stop_algo - start_algo) / double(CLOCKS_PER_SEC) * 1000;
//    cout << "Time taken by Kmeans Algorithm to segment the image " << TotalAlgoTime << endl;
//    // Create a new image with the assigned colors
//    Mat segmented=Mat::zeros(gray_img.size(), CV_32FC1);
//    for (int i = 0; i < gray_img.rows; i++) {
//        for (int j = 0; j < gray_img.cols; j++) {
//            int cluster = assignments[i * gray_img.cols + j];
//            segmented.at<float>(i, j) = centroids[cluster];
//        }
//    }   
//    //cout << gray_img.at<float>(300, 300) << endl;
//    //cout << segmented.at<float>(300, 300) << endl;
//    // Display the original and segmented images
//    cv::imshow("Original", gray_img);
//    cv::imshow("Segmented", segmented);
//    cv::waitKey(0);
//
//
//    
//
//    return 0;
//}
//
