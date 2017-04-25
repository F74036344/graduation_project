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
const int H_MIN = 0, H_MAX = 255;
const int S_MIN = 0, S_MAX = 255;
const int V_MIN = 0, V_MAX = 255;
//根據網路上查到的HSV膚色範圍: 0 < H < 50； 0.23 < S < 0.68 (跟Hue有較大的關係)
//int H_MIN_val = 0;
//int H_MAX_val = 70;
//int S_MIN_val = 58;
//int S_MAX_val = 174;
//int V_MIN_val = 40;
//int V_MAX_val = 255;
int H_MIN_val = 0;
int H_MAX_val = 70;
int S_MIN_val = 0;
int S_MAX_val = 231;
int V_MIN_val = 40;
int V_MAX_val = 255;

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

// 用來output點數的output file
ofstream outputfile;

// 用來進行去背處理的variable(定義在global，以讓function存取)
Ptr<BackgroundSubtractorMOG2> MOG2;


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
	// 1. 形狀辨識(background subtraction)
	// 先對所讀取到的影像進行去背處理
	Mat RoiFrame;
	Mat Mog_Mask, Mog_Mask_morpho, Mog_Mask_morpho_threshold;
	Mat backImg;

	blur(inputFrame, RoiFrame, Size(3, 3));
	//GaussianBlur(frame, RoiFrame, Size(9, 9), 0, 0);
	// Rect roi(100, 100, 300, 300);
	//RoiFrame = frame(roi);

	//Mog processing
	MOG2->apply(RoiFrame, Mog_Mask);

	//Background Image get
	MOG2->getBackgroundImage(backImg);

	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5), Point(4, 4));
	//morphology: 對Mog_Mask進行運算(CV_MOP_OPEN: erode侵蝕+dilate膨脹 )
	morphologyEx(Mog_Mask, Mog_Mask_morpho, CV_MOP_OPEN, Mat());
	morphologyEx(Mog_Mask_morpho, Mog_Mask_morpho, CV_MOP_ERODE, Mat());
	morphologyEx(Mog_Mask_morpho, Mog_Mask_morpho, CV_MOP_ERODE, Mat());
	morphologyEx(Mog_Mask_morpho, Mog_Mask_morpho, CV_MOP_DILATE, Mat(), Point(-1,-1), 7);

	//Binary: 尚須確認這在做什麼...
	threshold(Mog_Mask_morpho, Mog_Mask_morpho_threshold, 10, 255, CV_THRESH_BINARY);

	//imshow("ROI", RoiFrame);
	imshow("MogMask", Mog_Mask);
	imshow("BackImg", backImg);
	outputFrame = Mog_Mask_morpho_threshold.clone();

}

void extractSkinArea(const Mat& inputFrame, Mat& outputFrame/*, Mat& hsvImg, Mat& hsvThresImg, Mat& YCrCb*/) {

	// 用了兩種channel來進行顏色區段篩選
	// 1. HSV膚色範圍: 0 < H < 50； 0.23 < S < 0.68
	// HSV在面對強光時好像會讀不太到? -> 打算用YCrCb channel補強
	Mat hsv_img, hsv_skin_threshold_img;
	cvtColor(inputFrame, hsv_img, COLOR_BGR2HSV);
	inRange(hsv_img, 
		Scalar(H_MIN_val, S_MIN_val, V_MIN_val), 
		Scalar(H_MAX_val, S_MAX_val, V_MAX_val), 
		hsv_skin_threshold_img);
	// morphology(open運算 = erode運算 + dilate運算)
	morphologyEx(hsv_skin_threshold_img, hsv_skin_threshold_img, CV_MOP_OPEN, Mat());
	imshow("HSV", hsv_img);
	imshow("HSV Skin Threshold", hsv_skin_threshold_img);

	// 2. YCrCb膚色範圍: 135 < Cr < 180； 85 < Cb < 135； Y > 80
	Mat ycrcb_img, ycrcb_skin_threshold_img;
	cvtColor(inputFrame, ycrcb_img, COLOR_BGR2YCrCb);
	inRange(ycrcb_img, Scalar(80, 80, 85), Scalar(255, 180, 135), ycrcb_skin_threshold_img);
	// morphology
	morphologyEx(ycrcb_skin_threshold_img, ycrcb_skin_threshold_img, CV_MOP_OPEN, Mat());
	imshow("YCrCb", ycrcb_img);
	imshow("YCrCb Skin Threshold", ycrcb_skin_threshold_img);

	// 兩種方法取交集
	Mat resultSkinBin;
	bitwise_and(hsv_skin_threshold_img, ycrcb_skin_threshold_img, resultSkinBin);
	// Morphology
	morphologyEx(resultSkinBin, resultSkinBin, CV_MOP_OPEN, Mat());
	morphologyEx(resultSkinBin, resultSkinBin, CV_MOP_DILATE, Mat());

	outputFrame = resultSkinBin.clone();

	// Show the results:
	// imshow("Result", resultSkinBin);
}

int main() {

	VideoCapture camera;
	Mat frame;
	
	// 開檔，用來寫入手勢的點數
	outputfile.open("data.txt");
	// create slider bars for HSV filtering(這個是一開始用來測試HSV filter的數值用的，現在已經有找到不錯的數據所以暫時用不到了)
	createTrackbars();

	openCam(camera);	// 打開照相機

	// 產生去背景物件(history, varThreshold, detectShadows)
	MOG2 = createBackgroundSubtractorMOG2(2000, 64, true);

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

		// 1. Background Subtraction: Use MOG2 approach
		Mat Mog_Mask_morpho_threshold;
		backgroundSubtraction(frame, Mog_Mask_morpho_threshold);
		imshow("MOG2 Morpho Threshold", Mog_Mask_morpho_threshold);

		// 2. 色彩辨識法: 運用辨識皮膚顏色的方式來產生threshold
		Mat thresholdFrame;
		extractSkinArea(frame, thresholdFrame);
		imshow("Color Method Threshold", thresholdFrame);

		// 最後將兩者結果取交集(聯集?)
		Mat combinedThreshold;
		combinedThreshold = thresholdFrame.clone();
		// bitwise_not(Mog_Mask_morpho_threshold, Mog_Mask_morpho_threshold);
		bitwise_and(Mog_Mask_morpho_threshold, thresholdFrame, combinedThreshold);
		morphologyEx(combinedThreshold, combinedThreshold, CV_MOP_OPEN, Mat());
		imshow("Combined Threshold", combinedThreshold);

		// 接下來要將手的部位標示出來
		// Contour detection(red edge)
		Mat edges;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		findContours(combinedThreshold, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		size_t largestContour = 0;
		for (size_t i = 1; i < contours.size(); i++) {
			if (contourArea(contours[i]) > contourArea(contours[largestContour]))
				largestContour = i;
		}
		drawContours(frame, contours, largestContour, cv::Scalar(0, 0, 255), 1);

		// Convex hull(convex-set edge)
		if (!contours.empty()) {
			std::vector<std::vector<cv::Point> > hull(1);
			cv::convexHull(cv::Mat(contours[largestContour]), hull[0], false);
			cv::drawContours(frame, hull, 0, cv::Scalar(255, 130, 30), 3);
			// 根據convex set與contour之間的空隙來畫出convex defect(是這樣子稱呼嗎@@?)
			if (hull[0].size() > 2) {
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

		// 最後秀出處理的結果
		imshow("Result", frame);

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
