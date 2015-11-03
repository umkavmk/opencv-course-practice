#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void main()
{
    cv::Mat image = cv::imread("../road.png", CV_LOAD_IMAGE_COLOR);
    cv::imshow("original", image);
	cv::waitKey(2000);

	cv::Mat gray_image = cv::Mat(image.rows, image.cols, CV_8UC1);
    cv::cvtColor(image, gray_image, CV_BGR2GRAY);
    cv::imshow("after cvtColor", gray_image);
    cv::waitKey(1000);

	cv::Mat th_image = cv::Mat(image.rows, image.cols, CV_8UC1);
    cv::Canny(gray_image, th_image, 100, 200);
	cv::imshow("after Canny", th_image);
    cv::waitKey(1000);

	cv::Mat thInvert_image = cv::Mat(image.rows, image.cols, CV_8UC1);
    cv::bitwise_not(th_image, thInvert_image);
	cv::imshow("after bitwiseNot", thInvert_image);
    cv::waitKey(1000);

	cv::Mat distance_image = cv::Mat(image.rows, image.cols, CV_32F);
	distanceTransform(thInvert_image, distance_image, CV_DIST_L2, 3);
	cv::imshow("after distanceTransform", distance_image);
    cv::waitKey(1000);

	cv::Mat sum[3];//интегральные изображения
	cv::Mat chanels[3];//каналы

	split(image, chanels);//раскладываем на каналы

	for (int i = 0; i < 3; i++)
	{
		sum[i] = cv::Mat(image.rows + 1, image.cols + 1, CV_64F);
        integral(chanels[i], sum[i], CV_64F);
	}

    int size_rect;
    int x1, y1, x2, y2;
	double tmp_dist, tmp_sum;
	
	cv::Mat result = cv::Mat(image.rows, image.cols, CV_8UC3);
	
	for (int i = 0; i < distance_image.rows; i++)
	{
		for (int j = 0; j < distance_image.cols; j++)
		{
			tmp_dist = distance_image.at<float>(i, j);

			x1 = MAX((i - tmp_dist), 0);
			y1 = MAX((j - tmp_dist), 0);
			x2 = MIN((i + tmp_dist + 1), distance_image.rows);
			y2 = MIN((j + tmp_dist + 1), distance_image.cols);
			
			size_rect = (x2 - x1) * (y2 - y1);
			
			for (int k = 0; k < 3; k++)
			{
				if (size_rect == 0)
				{
					result.at<cv::Vec3b>(i, j)[k] = chanels[k].at<unsigned char>(i, j);
				}
				else
				{
					tmp_sum = sum[k].at<double>(x2, y2) - sum[k].at<double>(x1, y2)
						- sum[k].at<double>(x2, y1) + sum[k].at<double>(x1, y1);
					result.at<cv::Vec3b>(i, j)[k] = tmp_sum / size_rect;
				}
			}
		}
	}

    cv::imshow("result", result);
    cv::waitKey();
	cv::destroyAllWindows();

    return;
} 
