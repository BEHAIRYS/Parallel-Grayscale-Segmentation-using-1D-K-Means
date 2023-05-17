#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <Windows.h>
#include <tchar.h>
#include <random>   
#include<opencv2/imgproc/imgproc.hpp>
#include <mpi.h>

using namespace std;
using namespace cv;


int main()//int argc, char** argv)
{
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Start the timer
    double start_time = MPI_Wtime();

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cv::Mat gray_img, img;
    int k_centroids = 2;
    float* data = new float[270000];
    int* indexOfPoints = new int[270000];
    float* sum_points = new float[k_centroids * size * 2];
    float* centroids = new float[k_centroids];

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //cout << size << "\n";

    // Initialize data and centroids on all processes
    cv::imread("C:/Users/Mohasseb/Desktop/GrayscaleSegmentation.jpg", cv::IMREAD_COLOR).convertTo(img, CV_32FC3, (1. / 255.));
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

    int k = 0;
    for (int i = 0; i < gray_img.rows; i++) {
        for (int j = 0; j < gray_img.cols; j++) {
            data[k++] = gray_img.at<float>(i, j);
        }
    }

    // generate random k centroids 
    random_device rd; // create a random device object to seed the generator
    mt19937 gen(rd()); // create a Mersenne Twister engine object with seed from random_device
    uniform_int_distribution<int> distr(0, (gray_img.rows * gray_img.cols) - 1); // create a uniform integer distribution object with range [0, counter-1]

    //store the initial centroids 
    for (int i = 0; i < k_centroids; i++)
    {
        centroids[i] = data[distr(gen)];
    }

    // Broadcast centroids to all processes
    MPI_Bcast(centroids, k_centroids, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /*cout << rank << "\n";
    cout << k_centroids << "\n";
    for (int i = 0; i < k_centroids; i++) {
        cout << centroids[i] << "\n";
    }*/

    MPI_Barrier(MPI_COMM_WORLD);
    //// Calculate the chunk size for each process
    int chunk_size = (gray_img.rows * gray_img.cols) / size;
    // Scatter the image data to all processes
    float* chunk_data = new float[chunk_size];
    MPI_Scatter(data, chunk_size, MPI_FLOAT, chunk_data, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //cout << "tmam\n";

    bool converged = false;
    while (!converged) {

        int* centroidOfPoint = new int[chunk_size];
        float* sumCount = new float[k_centroids * 2];
        for (int i = 0; i < k_centroids * 2; i++)
            sumCount[i] = 0;

        for (int i = 0; i < chunk_size; i++) {
            float min_distance = FLT_MAX;
            int closest_centroid;
            for (int j = 0; j < k_centroids; j++) {
                float distance = abs(chunk_data[i] - centroids[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_centroid = j;
                }
            }
            centroidOfPoint[i] = closest_centroid;
            sumCount[closest_centroid * 2] += chunk_data[i];
            sumCount[closest_centroid * 2 + 1] += 1;
        }

        MPI_Gather(centroidOfPoint, chunk_size, MPI_INT, indexOfPoints, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(sumCount, k_centroids*2, MPI_FLOAT, sum_points, k_centroids*2, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            float* newCentroids = new float[k_centroids];
            for (int i = 0; i < k_centroids; i++) {
                float totalSumCentroid = 0;
                float totalCountCentroid = 0;
                for (int j = 0; j < size; j++) {
                    totalSumCentroid += sum_points[k_centroids * 2 * j + i*2];
                    totalCountCentroid += sum_points[k_centroids * 2 * j + i*2 + 1];
                }
                newCentroids[i] = totalSumCentroid / totalCountCentroid;
            }

             //Check for convergence
            double epsilon = 0.0001;
            converged = true;
            for (int i = 0; i < k_centroids; i++) {
                if (abs(newCentroids[i] - centroids[i]) > epsilon) {
                    converged = false;
                    break;
                }
            }
            centroids = newCentroids;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Stop the timer
    double end_time = MPI_Wtime();

    
    if (rank == 0) {
        
        // Create a new image with the assigned colors
        Mat segmented = Mat::zeros(gray_img.size(), CV_32FC1);
        for (int i = 0; i < gray_img.rows; i++) {
            for (int j = 0; j < gray_img.cols; j++) {
                int cluster = indexOfPoints[i * gray_img.cols + j];
                segmented.at<float>(i, j) = centroids[cluster];
            }
        }
        //cout << gray_img.at<float>(300, 300) << endl;
        //cout << segmented.at<float>(300, 300) << endl;
        // Display the original and segmented images
        cv::imshow("Original", gray_img);
        cv::imshow("Segmented", segmented);
        waitKey();
        cout << "finished\n";
        cout << "number of processes = " << size << endl;
        cout << "Execution time: " << end_time - start_time << " seconds" << endl;
    }
    
    MPI_Finalize();

    return 0;
}