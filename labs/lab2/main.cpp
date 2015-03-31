#include "cv.h" //main OpenCV functions
#include "highgui.h" //OpenCV GUI functions include <stdio.h>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

int main()
{
	VideoCapture cam("../temp.mp4");
	if (!cam.isOpened())
	{
		cout << "Can't connect to cam!" << endl;
	}

	Mat frame;
	Mat cur;
	Mat prev;
	namedWindow("opt_flow", 1);
	for(;;)
	{
		cam >> frame;
		if (frame.empty())
			break;

		cvtColor(frame, cur, CV_BGR2GRAY);

		Mat drawIm;
		frame.copyTo(drawIm);

		if (!prev.empty())
		{
			vector<Point2f> points;
			
			for (int i = 0; i < prev.rows; i = i+10)
				for (int j = 0; j < prev.cols; j = j+10)
					points.push_back(Point2f(j, i));
			
			vector<Point2f> resPoints;
			vector<uchar> status;
			vector<float>  err;

			Mat flow(cur.rows, cur.cols, CV_32FC2, Scalar(0, 0, 0));
			calcOpticalFlowFarneback(prev, cur, flow, 0.5, 1, 10, 3, 5, 1.1, 0);
			
			for (int i = 0; i < points.size(); i++)
			{
				const Point2f& fxy = flow.at<Point2f>(points[i].y, points[i].x);
				line(drawIm, points[i], Point(cvRound(points[i].x + fxy.x), cvRound(points[i].y + fxy.y)), Scalar(0, 0, 255), 2);
			}
		}

		imshow("opt_flow", drawIm);
		int key = cvWaitKey(1);     
		if( key==27 ) break;

		cur.copyTo(prev);
	}
}