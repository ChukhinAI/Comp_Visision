#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "stdlib.h"
#include <list>
using namespace std;
using namespace cv;
namespace GITGUDSON
{
	void find_contours(Mat src_gray)
	{
		blur(src_gray, src_gray, Size(3, 3));
		Mat canny_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		RNG rng(12345);
		/// Detect edges using canny
		Canny(src_gray, canny_output, 100, 200, 3);
		/// Find contours
		findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		/// Draw contours
		Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		}

		/// Show in a window
		namedWindow("Contours", WINDOW_AUTOSIZE);
		imshow("Contours", drawing);
	}

	void Task2()
	{
		Mat image, image2, image3, image4, image5;
		image = imread("b.png", IMREAD_GRAYSCALE);
		imshow("source", image);
		GaussianBlur(image, image, Size(5, 5), 0);
		auto rectangle = getStructuringElement(MORPH_RECT, Size(7, 7));
		morphologyEx(image, image2, MORPH_OPEN, rectangle);
		GaussianBlur(image2, image2, Size(5, 5), 0);
		morphologyEx(image2, image2, MORPH_OPEN, rectangle);
		auto ellipse = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
		//morphologyEx(image, image2, MORPH_OPEN, ellipse);
		find_contours(image2);
		//threshold(image2, image3, 200, 255, THRESH_BINARY);
		imshow("2", image2);
		//erode(image2, image3, getStructuringElement(MORPH_RECT, Size(6, 6)));
		//imshow("3", image3);
		find_contours(image2);
	}
}
int main(int argc, char** argv)
{
	GITGUDSON::Task2();
	waitKey();
	return EXIT_SUCCESS;
}