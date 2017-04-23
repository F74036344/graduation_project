#include <iostream>
// include commonly used
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/highgui.h>
#include <opencv/cv.h>

// For outputing to debu window
#include <windows.h>
#include <tchar.h>

#include <fstream>	// For outputing file
#include <sstream>	// Uh...
#include <string>


using namespace std;
using namespace cv;

//initial min and max HSV filter values.
//these will be changed using trackbars
const int H_MIN = 0, H_MAX = 500;
const int S_MIN = 0, S_MAX = 500;
const int V_MIN = 0, V_MAX = 500;
int H_MIN_val = 0;
int H_MAX_val = 17;
int S_MIN_val = 55;
int S_MAX_val = 175;
int V_MIN_val = 75;
int V_MAX_val = 400;

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
//names that will appear at the top of each window
const string cameraWindowName = "Original";
const string HSVWindowName = "HSV";
const string thresholdWindowName = "Threshold";
const string foregroundWindowName = "Foreground Image";
const string foregroundMaskWindowName = "Foreground Mask";
const string backgroundWindowName = "Mean Background Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

// 用來output點數的output file
ofstream outputfile;


void openCam(VideoCapture &camera)
{
	camera.open(0);
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

// What does this function do @@?
void on_trackbar(int, void*)
{//This function gets called whenever a
 // trackbar position is changed
	// So does this function do nothing?
}

void createTrackbars() {
	//create window for trackbars


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

int main() {

	/* --- 網路上的去背景的code --- */
	//VideoCapture cap(0); // open the default camera
	//if (!cap.isOpened()) // check if we succeeded
	//	return -1;


	//for (;;)
	//{

	//	cap >> frame; // get a new frame from camera
	//	if (frame.empty())
	//		break;

	//	//option 
	//	blur(frame(roi), RoiFrame, Size(3, 3));
	//	//RoiFrame = frame(roi);

	//	//Mog processing
	//	MOG2->apply(RoiFrame, Mog_Mask);

	//	if (LearningTime < 300)
	//	{
	//		LearningTime++;
	//		printf("background learning %d \n", LearningTime);
	//	}
	//	else
	//		LearningTime = 301;

	//	//Background Image get
	//	MOG2->getBackgroundImage(BackImg);

	//	//morphology 
	//	morphologyEx(Mog_Mask, Mog_Mask_morpho, CV_MOP_DILATE, element);
	//	//Binary
	//	threshold(Mog_Mask_morpho, Mog_Mask_morpho, 128, 255, CV_THRESH_BINARY);

	//	imshow("Origin", frame);
	//	imshow("ROI", RoiFrame);
	//	imshow("MogMask", Mog_Mask);
	//	imshow("BackImg", BackImg);
	//	imshow("Morpho", Mog_Mask_morpho);
	//}
	/* --- END of 網路上的去背景的code --- */
	
	// 開檔，用來寫入手勢的點數
	outputfile.open("data.txt");

	CascadeClassifier a;	// 
	VideoCapture camera;
	namedWindow(cameraWindowName, CV_WINDOW_AUTOSIZE);
	//namedWindow(HSVWindowName, CV_WINDOW_AUTOSIZE);
	//namedWindow(thresholdWindowName, CV_WINDOW_AUTOSIZE);
	//namedWindow(foregroundWindowName, CV_WINDOW_AUTOSIZE);
	//namedWindow(foregroundMaskWindowName, CV_WINDOW_AUTOSIZE);
	//namedWindow(backgroundWindowName, CV_WINDOW_AUTOSIZE);

	// 用來進行去背處理的variable
	Ptr<BackgroundSubtractorMOG2> MOG2 = createBackgroundSubtractorMOG2(3000, 64);
	//Options
	//MOG2->setHistory(3000);
	//MOG2->setVarThreshold(128);
	MOG2->setDetectShadows(false); //shadow detection on/off
	Mat Mog_Mask;
	Mat Mog_Mask_morpho;
	Mat Mog_Mask_morpho_threshold;

	// Rect roi(100, 100, 300, 300);

	namedWindow("ROI", CV_WINDOW_AUTOSIZE);
	Mat frame;
	Mat RoiFrame;
	Mat BackImg;

	int LearningTime = 0; //300;

	Mat element;
	element = getStructuringElement(MORPH_RECT, Size(9, 9), Point(4, 4));
	

	////create slider bars for HSV filtering
	//createTrackbars();

	// load(a); -> 這行的作用是....? 好像是用machine learning的方式來進行物體辨識(需載入xml檔)
	openCam(camera);	// 打開照相機

	// Infinite loop to acquire video frame continuously
	while (true)
	{
		// x and y values for the location of the object
		int x = 0, y = 0;

		// 將camera的影像存至frame
		if ( !camera.read(frame) )
		{
			std::cerr << "ERROR: Couldn't grab a camera frame." << std::endl;
			exit(1);
		}
		if (frame.empty())
			break;
		/* 想法: 
			1. 色彩辨識(Origin frame轉HSV，然後HSV再經過filter(inRange)轉binary)
			2. 形狀辨識(background subtraction)，用演算法判斷背景與主體，將背景去掉只留下主體的binary
			運用上面兩者所產生的threshold來取
		(原本是想用機器學習(haar cascade)來辨識，不過訓練失敗...，而且用內建的臉部辨識跑過，很耗CPU資源，暫不考慮)
		
		*/
		// 1. 
		// 先對所讀取到的影像進行去背處理
		//option 
		blur(frame, RoiFrame, Size(3, 3));
		//GaussianBlur(frame, RoiFrame, Size(9, 9), 0, 0);
		//RoiFrame = frame(roi);

		//Mog processing
		MOG2->apply(RoiFrame, Mog_Mask);

		if (LearningTime < 300)
		{
			LearningTime++;
			printf("background learning %d \n", LearningTime);
		}
		else
			LearningTime = 301;

		//Background Image get
		MOG2->getBackgroundImage(BackImg);

		//morphology 
		morphologyEx(Mog_Mask, Mog_Mask_morpho, CV_MOP_DILATE, element);
		//Binary
		threshold(Mog_Mask_morpho, Mog_Mask_morpho_threshold, 128, 255, CV_THRESH_BINARY);

		//// 先對所讀取到的影像進行去背處理(old version)
		//// Grab the next camera frame.
		//Mat frame;
		//Mat fgMOG2MaskImg, fgMOG2Img, bgMOG2Img;	// fgMOG2MaskImg: 計算後的背景影像；fgMOG2Img: 當下擷取的影像
		//Mat HSVFrame;
		//Mat thresholdFrame;
		//{
		//	int history = 500;
		//	double varThreshold = 16;
		//	bool detectShadows = true;
		//	cv::Ptr<cv::BackgroundSubtractor> pMOG2;
		//	pMOG2 = cv::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows);
		//	bool update_bg_model = true;
		//	if (fgMOG2Img.empty())
		//		fgMOG2Img.create(frame.size(), frame.type());
		//	// update the model
		//	pMOG2->apply(frame, fgMOG2MaskImg, update_bg_model ? -1 : 0);
		//	fgMOG2Img = cv::Scalar::all(0);
		//	frame.copyTo(fgMOG2Img, fgMOG2MaskImg);

		//	pMOG2->getBackgroundImage(bgMOG2Img);
		//	cv::imshow(foregroundMaskWindowName, fgMOG2MaskImg);
		//	cv::imshow(foregroundWindowName, fgMOG2Img);
		//	if (!bgMOG2Img.empty())
		//		cv::imshow(backgroundWindowName, bgMOG2Img);
		//}
		
		// 2. 色彩辨識法: 運用辨識皮膚顏色的方式，
		// 影像的轉換(->HSV->threshold matrix->contour detection->convex hull)，方便辨識
		// convert frame from BGR to HSV colorspace
		Mat HSVFrame;
		Mat thresholdFrame;
		cvtColor(frame, HSVFrame, COLOR_BGR2HSV);
		// threshold matrix
		inRange(HSVFrame, Scalar(H_MIN_val, S_MIN_val, V_MIN_val), Scalar(H_MAX_val, S_MAX_val, V_MAX_val), thresholdFrame);
		int blurSize = 5;
		int elementSize = 5;
		medianBlur(thresholdFrame, thresholdFrame, blurSize);


		// Contour detection(red edge)
		Mat edges;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		findContours(Mog_Mask_morpho_threshold, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		size_t largestContour = 0;
		for (size_t i = 1; i < contours.size(); i++)
		{
			if (contourArea(contours[i]) > contourArea(contours[largestContour]))
				largestContour = i;
		}
		drawContours(frame, contours, largestContour, cv::Scalar(0, 0, 255), 1);

		// Convex hull(convex-set edge)
		if (!contours.empty())
		{
			std::vector<std::vector<cv::Point> > hull(1);
			cv::convexHull(cv::Mat(contours[largestContour]), hull[0], false);
			cv::drawContours(frame, hull, 0, cv::Scalar(255, 130, 30), 3);
			// 根據convex set與contour之間的空隙來畫出convex defect(是這樣子稱呼嗎@@?)
			if (hull[0].size() > 2)
			{
				std::vector<int> hullIndexes;
				cv::convexHull(cv::Mat(contours[largestContour]), hullIndexes, true);
				std::vector<cv::Vec4i> convexityDefects;
				cv::convexityDefects(cv::Mat(contours[largestContour]), hullIndexes, convexityDefects);
				cv::Rect boundingBox = cv::boundingRect(hull[0]);
				cv::rectangle(frame, boundingBox, cv::Scalar(255, 0, 0));
				cv::Point center = cv::Point(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
				std::vector<cv::Point> validPoints;
				for (size_t i = 0; i < convexityDefects.size(); i++)
				{
					cv::Point p1 = contours[largestContour][convexityDefects[i][0]];
					cv::Point p2 = contours[largestContour][convexityDefects[i][1]];
					cv::Point p3 = contours[largestContour][convexityDefects[i][2]];
					double angle = std::atan2(center.y - p1.y, center.x - p1.x) * 180 / CV_PI;
					double inAngle = innerAngle(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
					double length = std::sqrt(std::pow(p1.x - p3.x, 2) + std::pow(p1.y - p3.y, 2));
					if (angle > -30 && angle < 160 && std::abs(inAngle) > 20 && std::abs(inAngle) < 120 && length > 0.1 * boundingBox.height) {
						validPoints.push_back(p1);
					}
				}
				for (size_t i = 0; i < validPoints.size(); i++) {
					cv::circle(frame, validPoints[i], 10, cv::Scalar(255, 130, 30), 2);
				}
				// 將valid point寫入outputfile
				outputfile << validPoints.size() << " ";
				// 將valid point的數量print到frame影像上
				putText(frame, format("%d", validPoints.size()), Point(50, 50), 1, 4, Scalar(0, 0, 255), 2);

			}
		}

		// 最後秀出各階段處理的結果
		imshow(cameraWindowName, frame);
		imshow("ROI", RoiFrame);
		imshow("MogMask", Mog_Mask);
		imshow("BackImg", BackImg);
		imshow("Morpho", Mog_Mask_morpho);
		imshow("Morpho Threshold", Mog_Mask_morpho_threshold);

		//// Use morph (erode and dilate the thresholdFrame)
		//morphOps(thresholdFrame);
		//// ?
		//// trackFilteredObject(x, y, thresholdFrame, frame);

		//// Show frames
		//imshow(cameraWindowName, frame);
		//imshow(HSVWindowName, HSVFrame);
		//imshow(thresholdWindowName, thresholdFrame);

		// printf("%d\n", cv::waitKey(msPerFrame));
		if (cv::waitKey(msPerFrame) != 255) {
			// 如果有輸入任意按鍵 -> 退出
			exitProgram();			
		}	
	}

	return 1;
}
