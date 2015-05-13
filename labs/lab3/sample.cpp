#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

struct Object
{
	Rect bb;
	int non_detected_time;
	int life_time;
};


bool isIntersec(Rect r, Object obj)
{
	return true;
}

int main()
{
	CascadeClassifier cascade("haarcascade_frontalface_alt.xml");

	VideoCapture cam("../gymn.avi");
	if (!cam.isOpened())
	{
		cout << "Can't connect to cam or file!" << endl;
	}
	Mat frame;
	Mat prev;
	namedWindow("src", 1);
	vector<Object> objects;
	for(;;)
	{
		cam >> frame;

		if (frame.empty())
			break;

		vector<Rect> rects;
		cascade.detectMultiScale(frame, rects, 1.1, 5, 0, Size(50, 50));
		for(int i = 0; i < rects.size(); i++)
		{
			rectangle(frame, rects[i], cv::Scalar(255,0,0));
		}
		imshow("src", frame);
		int key = cvWaitKey(1);
		if (key==27) break;

		prev = frame.clone();
	}
}