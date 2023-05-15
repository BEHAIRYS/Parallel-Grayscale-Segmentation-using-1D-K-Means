#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include <iostream>
#include <Windows.h>
#include <tchar.h>
#include <random>   
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{

   
    cv::Mat img;
    cv::Mat gray_img;
    cv::imread("C:/Users/HP/Desktop/two.jpg", cv::IMREAD_COLOR).convertTo(img, CV_32FC3, (1. / 255.));
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    namedWindow("First OpenCV Application", WINDOW_AUTOSIZE);
    cv::imshow("First OpenCV Application", gray_img);
    cv::moveWindow("First OpenCV Application", 0, 45);
    cv::waitKey(0);
    cv::destroyAllWindows();

   
    // constructing 1D array of pixel values of the image
    float* data = new float[gray_img.rows * gray_img.cols * gray_img.channels()];
    
    //cout << gray_img.at<float>(2,1) << endl;
    int counter = 0;
         #pragma omp for
         for (int i = 0; i < img.rows; i++) 
         {  
            for (int j = 0; j < img.cols; j++) {
                data[counter] = gray_img.at<float>(i, j);
                //cout << data[counter] << " ";
                //cout << tid <<" "<<counter << endl;
                #pragma omp atomic
                counter++;
            }
         }


         // generate random 3 centroids 
         random_device rd; // create a random device object to seed the generator
         mt19937 gen(rd()); // create a Mersenne Twister engine object with seed from random_device
         uniform_int_distribution<int> distr(0, counter - 1); // create a uniform integer distribution object with range [0, counter-1]
         //cout << counter<< endl;
         int k = 4;
         vector<float> centroids(k);
        #pragma omp single
         {
             //store the initial centroids 
             for (int i = 0; i < k; i++)
             {
                 centroids[i] = data[distr(gen)];
                 //cout << distr(gen) << " ";
                 //cout << centroids[i]<<endl;
             }
         }

    vector<int> assignment(counter);
    bool converged = false;
    while (!converged) {
        // Assign each pixel to the closest centroid
        vector<int> assignments(counter);
        #pragma omp for
        for (int i = 0; i < counter; i++) {
            float min_distance = FLT_MAX;
            int closest_centroid;
            for (int j = 0; j < k; j++) {
                float distance = norm(data[i] - centroids[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_centroid = j;
                }
            }
            assignments[i] = closest_centroid;
        }
        vector<float> new_centroids(k);
        vector<int> counts(k, 0);
        #pragma omp parallel for
        for (int i = 0; i < counter; i++) {
            int cluster = assignments[i];
            new_centroids[cluster] += data[i];
            counts[cluster]++;
        }
        #pragma omp for
        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) {
                new_centroids[i] /= counts[i];
            }
        }



        // Check for convergence
        float epsilon = 0.01;
        converged = true;
        #pragma omp for
        for (int i = 0; i < k; i++) {
            if (norm(new_centroids[i] - centroids[i]) > epsilon) {
                converged = false;
                break;
            }
        }
        #pragma omp barrier
        //update the centroids
        #pragma omp single
        {
            centroids = new_centroids;
            assignment = assignments;
        }
        
        
    }
    cout << centroids[0] << " " << centroids[1] << " " << centroids[2] << " " << endl;
    // Create a new image with the assigned colors
    Mat segmented=Mat::zeros(gray_img.size(), CV_32FC1);
    for (int i = 0; i < gray_img.rows; i++) {
        for (int j = 0; j < gray_img.cols; j++) {
            int cluster = assignment[i * gray_img.cols + j];
            segmented.at<float>(i, j) = centroids[cluster];
        }
    }   
    //cout << gray_img.at<float>(300, 300) << endl;
    //cout << segmented.at<float>(300, 300) << endl;
    // Display the original and segmented images
    cv::imshow("Original", gray_img);
    cv::imshow("Segmented", segmented);
    waitKey();


    

    return 0;
}
