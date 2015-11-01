#include <opencv2\opencv.hpp>

int main()
{
	cv::Mat image1 = cv::imread("findcontours.png", 0);
	
	cv::imshow("distance - original", image1);
	cv::waitKey(2000);

	cv::Mat image1_copy;

	cv::distanceTransform(image1, image1_copy, CV_DIST_L2, 3);
	cv::normalize(image1_copy, image1_copy, 0.0, 1.0, cv::NORM_MINMAX);

	cv::imshow("distance - result", image1_copy);
	cv::waitKey(2000);
	
	cv::Mat image2 = cv::imread("house.png", 1);

	cv::imshow("blur - original", image2);
	cv::waitKey(2000);
	

	cv::Mat image2_copy = image2.clone();
	
	cv::blur(image2, image2_copy, cv::Size(5,5));

	cv::imshow("blur - result", image2_copy);
	cv::waitKey(2000);

	cv::Mat image3 = cv::imread("lena.jpg", 1);
	cv::imshow("canny - original", image3);
	cv::waitKey(2000);

	cv::Mat image3_copy = image3.clone();
	cv::Canny(image3, image3_copy, 0.0, 5.0);

	cv::imshow("canny - result", image3_copy);
	cv::waitKey(2000);
	return 0;
}
