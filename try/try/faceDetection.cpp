#include<opencv2/objdetect.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

#include<iostream>
#include<stdio.h>

using namespace std;
using namespace cv;

string faceCascadePath;
CascadeClassifier faceCascade;

int main() {
	faceCascadePath = "haarcascade_frontalface_default.xml";
	if (!faceCascade.load(faceCascadePath)) {
		printf("--(!)Error Loading Cascade\n");
		return -1;
	}
	VideoCapture cam(0);
	if (!cam.isOpened()) { 
		cout << "fail to open camera\n";
		return -1;
	}
	int frame_width = cam.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = cam.get(CAP_PROP_FRAME_HEIGHT);
	vector<Rect> faces;

	while (true) {
		Mat frame;
		cam >> frame;
		Mat frameGray;
		cvtColor(frame, frameGray, COLOR_BGR2GRAY);

		//	frameClone = frame.clone();
		faceCascade.detectMultiScale(frameGray, faces, 1.2, 3);
			for (size_t i = 0; i < faces.size(); i++) {
				int x = faces[i].x;
				int y = faces[i].y;
				int w = faces[i].width;
				int h = faces[i].height;
				rectangle(frame, Point(x, y), Point(x + w, y + h), Scalar(255, 0, 0), 2, 4);
			}
			//	putText(frameClone, format("# Neighbors = %d", neigh), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 4);
		imshow("Face Detection Demo", frame);
		int k = waitKey(25);
		if (k == 27) break;
	}
		cam.release();
		destroyAllWindows();
}