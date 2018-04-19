#include<opencv2/objdetect.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

#include<iostream>
#include<stdio.h>

using namespace std;
using namespace cv;

string faceCascadePath;
string smileCascadePath;
CascadeClassifier faceCascade;
CascadeClassifier smileCascade;

int main() {
	faceCascadePath = "haarcascade_frontalface_default.xml";
	smileCascadePath = "haarcascade_smile.xml";
	if (!faceCascade.load(faceCascadePath)) {
		printf("--(!)Error Loading Cascade\n");
		return -1;
	}
	if (!smileCascade.load(smileCascadePath)) {
		cout << "fail to load smile cascade\n";
	}
	vector<Rect> faces;
	vector<Rect> smiles;

	while (true) {
		Mat frame = imread("hillary_clinton.jpg");
		Mat frameGray;
		cvtColor(frame, frameGray, COLOR_BGR2GRAY);

		//	frameClone = frame.clone();
		faceCascade.detectMultiScale(frameGray, faces, 1.2, 21);

			for (size_t i = 0; i < faces.size(); i++) {
				int x = faces[i].x;
				int y = faces[i].y;
				int w = faces[i].width;
				int h = faces[i].height;
				rectangle(frame, Point(x, y), Point(x + w, y + h), Scalar(255, 0, 0), 2, 4);
				
				Mat faceROI = frameGray(faces[i]);
				smileCascade.detectMultiScale(faceROI, smiles, 1.2, 256);
				for (size_t j = 0; j < smiles.size(); j++) {
					int smileX = smiles[j].x;
					int smileY = smiles[j].y;
					int smileW = smiles[j].width;
					int smileH = smiles[j].height;
					rectangle(frame, Point(smileX+x, smileY+y), Point(smileX+x + smileW, smileY+y + smileH), Scalar(0, 255, 0), 2, 4);

				}
			}
			//	putText(frameClone, format("# Neighbors = %d", neigh), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 4);
		imshow("Face Detection Demo", frame);
		int k = waitKey(25);
		if (k == 27) break;
	}
		destroyAllWindows();
}