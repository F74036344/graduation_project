// version: 17.11.6

// include opencv lib
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/highgui.h>
#include <opencv/cv.h>

// For outputing to debug window
#include <windows.h>
#include <tchar.h>

// General C/C++ lib
#include <iostream>
#include <fstream>	// For outputing file
#include <sstream>
#include <string>
#include <climits>


using namespace std;
using namespace cv;


// Video to test
const string videoName = "sunrise.mp4";

// Initial min and max HSV filter values.
// These could be changed by using trackbars
const int H_MIN = 0, H_MAX = 255;
const int S_MIN = 0, S_MAX = 255;
const int V_MIN = 0, V_MAX = 255;

// 用來存trackbar的初始HSV值的var 
int H_MIN_val = 0;
int H_MAX_val = 70;
int S_MIN_val = 58;
int S_MAX_val = 174;
int V_MIN_val = 40;
int V_MAX_val = 255;

//int H_MIN_val = 0;
//int H_MAX_val = 70;
//int S_MIN_val = 0;
//int S_MAX_val = 231;
//int V_MIN_val = 40;
//int V_MAX_val = 255;

int msPerFrame = 30;

// 12,74,14,95,132,202
//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = FRAME_HEIGHT / 1.333;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 15 * 15;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;

// 用來進行去背處理的variable(定義在global，以讓function存取)
Ptr<BackgroundSubtractorMOG2> MOG2;


void openCam(VideoCapture &camera)
{
	// camera.open( "filepath" );
	camera.open( "../test_case/fire1.mp4" );	// Opencv的imread或VideoCapture物件會以"BGR"的方式儲存，而非RGB
	if (!camera.isOpened())
	{
		std::cout << "Camera CANNOT open";
		exit(EXIT_FAILURE);
	}

	// 設定顯示影像的window的大小
	camera.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
}

void exitProgram() {
	// At last, exit with succesful state
	exit(EXIT_SUCCESS);
}

void on_trackbar(int, void*)
{
	//This function gets called whenever a
	// trackbar position is changed
	// 在trackbar移動的時候沒有任何需要做的事情，所以留空
}

void createTrackbars() {
	// 產生用來測試用的視窗

	// 為視窗命名一個名稱
	const string trackbarWindowName = "Trackbar Window";
	namedWindow(trackbarWindowName, 0);
	//create memory to store trackbar name on window
	//char TrackbarName[50];
	//sprintf(TrackbarName, "H_MIN");
	//sprintf(TrackbarName, "H_MAX");
	//sprintf(TrackbarName, "S_MIN");
	//sprintf(TrackbarName, "S_MAX");
	//sprintf(TrackbarName, "V_MIN");
	//sprintf(TrackbarName, "V_MAX");

	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH), 
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->      
	createTrackbar("H_MIN", trackbarWindowName, &H_MIN_val, H_MAX, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &H_MAX_val, H_MAX, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &S_MIN_val, S_MAX, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &S_MAX_val, S_MAX, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &V_MIN_val, V_MAX, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &V_MAX_val, V_MAX, on_trackbar);

}

void drawObject(int x, int y, Mat &frame) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25>0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25<FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - 25>0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25<FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, std::to_string(x) + "," + std::to_string(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);

}

void morphOps(Mat &thresh) {

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(6, 6));

	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);

	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);
}

// 
void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects<MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if (area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
					x = moment.m10 / area;
					y = moment.m01 / area;
					objectFound = true;
					refArea = area;
				}
				else objectFound = false;


			}
			//let user know you found an object
			if (objectFound == true) {
				putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
				//draw object location on screen
				drawObject(x, y, cameraFeed);
			}

		}
		else putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
	}
}

float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1)
{
	float dist1 = std::sqrt((px1 - cx1)*(px1 - cx1) + (py1 - cy1)*(py1 - cy1));
	float dist2 = std::sqrt((px2 - cx1)*(px2 - cx1) + (py2 - cy1)*(py2 - cy1));

	float Ax, Ay;
	float Bx, By;
	float Cx, Cy;

	//find closest point to C  
	//printf("dist = %lf %lf\n", dist1, dist2);  
	Cx = cx1;
	Cy = cy1;
	if (dist1 < dist2) {
		Bx = px1;
		By = py1;
		Ax = px2;
		Ay = py2;
	}
	else {
		Bx = px2;
		By = py2;
		Ax = px1;
		Ay = py1;
	} 

	float Q1 = Cx - Ax;
	float Q2 = Cy - Ay;
	float P1 = Bx - Ax;
	float P2 = By - Ay;

	float A = std::acos((P1*Q1 + P2*Q2) / (std::sqrt(P1*P1 + P2*P2) * std::sqrt(Q1*Q1 + Q2*Q2)));
	A = A * 180 / CV_PI;
	return A;
}

void backgroundSubtraction(const Mat& inputFrame, Mat& outputFrame) {
	// 對input影像進行去背處理
	Mat RoiFrame;
	Mat Mog_Mask, Mog_Mask_morpho, Mog_Mask_morpho_threshold;
	Mat backImg;

	blur(inputFrame, RoiFrame, Size(3, 3));
	// GaussianBlur(frame, RoiFrame, Size(9, 9), 0, 0);
	// Rect roi(100, 100, 300, 300);
	// RoiFrame = frame(roi);
	// 
	// Mog processing
	MOG2->apply(RoiFrame, Mog_Mask);

	// Background Image get
	MOG2->getBackgroundImage(backImg);

	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5), Point(4, 4));
	// morphology: 對Mog_Mask進行運算(CV_MOP_OPEN: erode侵蝕+dilate膨脹 )
	morphologyEx(Mog_Mask, Mog_Mask_morpho, CV_MOP_OPEN, Mat());
	//morphologyEx(Mog_Mask_morpho, Mog_Mask_morpho, CV_MOP_ERODE, Mat());
	//morphologyEx(Mog_Mask_morpho, Mog_Mask_morpho, CV_MOP_ERODE, Mat());
	//morphologyEx(Mog_Mask_morpho, Mog_Mask_morpho, CV_MOP_DILATE, Mat(), Point(-1,-1), 7);

	// Binary: 尚須確認這在做什麼...
	threshold(Mog_Mask_morpho, Mog_Mask_morpho_threshold, 10, 255, CV_THRESH_BINARY);

	// imshow("ROI", RoiFrame);
	// imshow("MogMask", Mog_Mask);
	imshow("背景圖", backImg);
	outputFrame = Mog_Mask_morpho_threshold.clone();

}

void extractFireArea(const Mat& inputFrame, Mat& outputFrame/*, Mat& hsvImg, Mat& hsvThresImg, Mat& YCrCb*/) {

	// inputFrame為 BGR channel

	// 使用YCrCb channel來協助提取火焰像素(Y: 流明;灰階值；Cr: 紅色偏移量; Cb: 藍色偏移量)
	Mat inputFrame_YCrCb, fire_YCrCb_threshold, resultFireBin;
	Mat tempThres(inputFrame.rows, inputFrame.cols, CV_8U);
	// 首先，將輸入的影像從BGR轉成YCrCb，然後另外存在inputFrame_YCrCb
	cvtColor(inputFrame, inputFrame_YCrCb, COLOR_BGR2YCrCb);

	// **然後根據條件，將火焰像素提取出來
	int pixelAmt = inputFrame.rows * inputFrame.cols;
	
	uchar *pxptr_BGR, *pxptr_YCrCb, *pxptr_temp;
	// 走訪每一個pixel，並檢驗每個pixel是否有符合條件，有則判定為火焰像素
	// 先算出mean值
	float b_mean = 0, g_mean = 0, r_mean = 0;
	float y_mean = 0, cr_mean = 0, cb_mean = 0;

	for (int i = 0; i < inputFrame.rows; i++) {
		pxptr_YCrCb = (uchar*)inputFrame_YCrCb.ptr<uchar>(i); // Point to first pixel in the row of YCrCb frame
		pxptr_BGR = (uchar*)inputFrame.ptr<uchar>(i);

		for (int j = 0; j < inputFrame.cols; j++) {
			b_mean += *pxptr_BGR;	*pxptr_BGR++;
			g_mean += *pxptr_BGR;	*pxptr_BGR++;
			r_mean += *pxptr_BGR;	*pxptr_BGR++;
			y_mean += *pxptr_YCrCb;		*pxptr_YCrCb++;
			cr_mean += *pxptr_YCrCb;	*pxptr_YCrCb++;
			cb_mean += *pxptr_YCrCb;	*pxptr_YCrCb++;

		}
	}
	b_mean /= pixelAmt;
	g_mean /= pixelAmt;
	r_mean /= pixelAmt;
	y_mean /= pixelAmt;
	cr_mean /= pixelAmt;
	cb_mean /= pixelAmt;

	for (int i = 0; i < inputFrame.rows; i++) {
		pxptr_YCrCb = (uchar*)inputFrame_YCrCb.ptr<uchar>(i); // Point to first pixel in the row of YCrCb frame
		pxptr_BGR = (uchar*)inputFrame.ptr<uchar>(i);
		pxptr_temp = (uchar*)tempThres.ptr<uchar>(i); // Point to first pixel in the row of tempThres frame

		for (int j = 0; j < inputFrame.cols; j++) {
			uchar b = *pxptr_BGR++;
			uchar g = *pxptr_BGR++;
			uchar r = *pxptr_BGR++;
			uchar y = *pxptr_YCrCb++;
			uchar cr = *pxptr_YCrCb++;
			uchar cb = *pxptr_YCrCb++;


			if ( (y >= cb && cr >= cb) // 根據"2012_07_18_大本論文_廖翌涵_圖書館"的條件
				 && ( y >= y_mean && cr >= cr_mean && cb <= cb_mean ) 
				 && ( r >= g && g >= b )
					&& ( r >= r_mean && g >= g_mean && b >= b_mean)
				) {	// 根據jeans_1115_..
				*pxptr_temp = UCHAR_MAX;	// 設為白色
			}
			else {	// Not fire pixel
				*pxptr_temp = 0;	// 設為黑色
			}
			pxptr_temp++;
		}
	}




	//// 使用OpenCV為Mat提供的迭代器(與STL迭代器兼容)，速度較與用at取值來的快
	//Mat_<Vec3b>::iterator it = inputFrame_YCrCb.begin<Vec3b>();
	//Mat_<Vec3b>::iterator it_end = inputFrame_YCrCb.end<Vec3b>();
	//Mat_<uchar>::iterator it2 = tempThres.begin<uchar>();
	//Mat_<uchar>::iterator it2_end = tempThres.end<uchar>();
	//for (; it != it_end; it++, it2++) {
	//	// channel number: Y:0, Cr:1, Cb:2 
	//	// if Y > Cb && Cr > Cb，則判定為火焰像素
	//	if ((*it)[0] > (*it)[2] && (*it)[1] > (*it)[2]) {
	//		(*it2) = UCHAR_MAX;	// 火焰像素 -> 設為白色
	//	}
	//	else {
	//		(*it2) = 0;	// 非火焰像素 -> 設為黑色
	//	}
	//}

	// inRange(inputFrame_YCrCb, Scalar(80, 80, 85), Scalar(255, 180, 135), fire_YCrCb_threshold);
	// morphology
	// morphologyEx(fire_YCrCb_threshold, fire_YCrCb_threshold, CV_MOP_OPEN, Mat());
	// 顯示inputFrame_YCrCb(debug用)
	imshow("YCrCb", inputFrame_YCrCb);

	// Morphology
	//morphologyEx(tempThres, resultFireBin, CV_MOP_OPEN, Mat());
	//morphologyEx(resultFireBin, resultFireBin, CV_MOP_DILATE, Mat());

	outputFrame = tempThres.clone();

	// Show the results:
	// imshow("Result", resultFireBin);
}

int main() {

	VideoCapture camera;
	Mat frame;
	
	// create slider bars for HSV filtering(這個是一開始用來測試HSV filter的數值用的，現在已經有找到不錯的數據所以暫時用不到了)
	// createTrackbars();

	openCam(camera);	// 打開照相機

	// 產生去背景物件(history, varThreshold, detectShadows)
	MOG2 = createBackgroundSubtractorMOG2(20000, 64, true);

	// Infinite loop to acquire video frame continuously
	while (true)
	{
		// x and y values for the location of the object
		int x = 0, y = 0;

		// 將camera的影像存至frame
		if ( !camera.read(frame) || frame.empty() ) {
			std::cerr << "ERROR: Couldn't grab a camera frame." << std::endl;
			exit(EXIT_FAILURE);
		}
		/* 想法: 
			1. 色彩辨識: Origin frame轉HSV，然後HSV再經過filter(inRange)選取接近皮膚的顏色，然後轉binary
			2. 形狀辨識(background subtraction): 用演算法判斷背景與主體，將背景去掉只留下主體的binary
			運用上面兩者所產生的threshold取交集來留下皮膚色的部分，然後再與background subtraction取交集。
			得到的結果再根據形狀去判斷手勢。(不過目前還有一些問題，例如會連同臉一起讀進去...)
			(原本是想用機器學習(haar cascade)來辨識，不過訓練失敗...
			而且用內建的臉部辨識跑過，也上網查過資料，發現haar cascade很耗CPU資源，暫不考慮)
		流程:
			->color extraction
			->background subtraction
			->threshold matrix
			->contour detection
			->convex hull
		*/

		// 1. Background Subtraction: Use MOG2 approach(MOG2: 一種基於"混和高斯模型"的演算法)
		// 先進行去背景處理(靜止很久的物件會被歸類為背景)
		Mat Mog_Mask_morpho_threshold;
		backgroundSubtraction(frame, Mog_Mask_morpho_threshold);	// 出來的影像會是二值圖
		imshow("經過MOG2去背景處理後的影像", Mog_Mask_morpho_threshold);

		// 2. 色彩辨識法: 運用辨識火焰顏色的方式來產生threshold
		// 判斷條件: 根據經驗法則
		Mat thresholdFrame;
		extractFireArea(frame, thresholdFrame);	// 出來的影像會是二值圖
		imshow("Fire Area Extracted by color method (YCrCb)", thresholdFrame);

		// 最後將兩者結果取交集
		Mat combinedThreshold = thresholdFrame.clone();
		bitwise_and(Mog_Mask_morpho_threshold, thresholdFrame, combinedThreshold);
		morphologyEx(combinedThreshold, combinedThreshold, CV_MOP_OPEN, Mat());	// 進行開運算
		imshow("Combined Threshold", combinedThreshold);

		// 得到火焰區塊的二值圖後，接下來要將火焰的部分用線標示出來
		// Contour detection(red edge)
		Mat edges;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		findContours(combinedThreshold, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		size_t largestContour = 0;
		for (size_t i = 1; i < contours.size(); i++) {
			drawContours(frame, contours, i, cv::Scalar(255, 180, 0), 2);
		}

		// Old code: 只畫出最大的contours
		//for (size_t i = 1; i < contours.size(); i++) {
		//	if (contourArea(contours[i]) > contourArea(contours[largestContour]))
		//		largestContour = i;
		//}
		//drawContours(frame, contours, largestContour, cv::Scalar(0, 0, 255), 2);

		//再更進一步的標示出來(不過還不知道這個是如何標示的)

		// Convex hull(convex-set edge)
		//if (!contours.empty()) {
		//	std::vector<std::vector<cv::Point> > hull(1);
		//	cv::convexHull(cv::Mat(contours[largestContour]), hull[0], false);
		//	cv::drawContours(frame, hull, 0, cv::Scalar(255, 130, 30), 3);
		//	// 根據convex set與contour之間的空隙來畫出convex defect(是這樣子稱呼嗎@@?)
		//	if (hull[0].size() > 2) {
		//		std::vector<int> hullIndexes;
		//		cv::convexHull(cv::Mat(contours[largestContour]), hullIndexes, true);
		//		std::vector<cv::Vec4i> convexityDefects;
		//		cv::convexityDefects(cv::Mat(contours[largestContour]), hullIndexes, convexityDefects);
		//		cv::Rect boundingBox = cv::boundingRect(hull[0]);
		//		cv::rectangle(frame, boundingBox, cv::Scalar(255, 0, 0));
		//		cv::Point center = cv::Point(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
		//		std::vector<cv::Point> validPoints;
		//		for (size_t i = 0; i < convexityDefects.size(); i++)
		//		{
		//			cv::Point p1 = contours[largestContour][convexityDefects[i][0]];
		//			cv::Point p2 = contours[largestContour][convexityDefects[i][1]];
		//			cv::Point p3 = contours[largestContour][convexityDefects[i][2]];
		//			double angle = std::atan2(center.y - p1.y, center.x - p1.x) * 180 / CV_PI;
		//			double inAngle = innerAngle(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
		//			double length = std::sqrt(std::pow(p1.x - p3.x, 2) + std::pow(p1.y - p3.y, 2));
		//			if (angle > -30 && angle < 160 && std::abs(inAngle) > 20 && std::abs(inAngle) < 120 && length > 0.1 * boundingBox.height) {
		//				validPoints.push_back(p1);
		//			}
		//		}
		//		for (size_t i = 0; i < validPoints.size(); i++) {
		//			cv::circle(frame, validPoints[i], 10, cv::Scalar(255, 130, 30), 2);
		//		}
		//		
		//	}
		//}

		// 最後秀出處理的結果
		imshow("Result", frame);

		//// Use morph (erode and dilate the thresholdFrame)
		//morphOps(thresholdFrame);
		//// ?
		//// trackFilteredObject(x, y, thresholdFrame, frame);

		//// Show frames
		//imshow(cameraWindowName, frame);
		//imshow(thresholdWindowName, thresholdFrame);

		// printf("%d\n", cv::waitKey(msPerFrame));
		if (cv::waitKey(msPerFrame) != 255) {
			// 如果有輸入任意按鍵 -> 退出
			exitProgram();			
		}

	}

	return 1;
}
