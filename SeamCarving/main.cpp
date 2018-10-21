#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void avgPixel(Vec3b v1, Vec3b v2, Vec3b& res) {
	res[0] = (v1[0] + v2[0]) / 2;
	res[1] = (v1[1] + v2[1]) / 2;
	res[2] = (v1[2] + v2[2]) / 2;
}

Mat energe(Mat image) {
	int rows = image.rows, cols = image.cols;
	Mat energy_image, dx, dy, grad, abs_dx, abs_dy;

	GaussianBlur(image, energy_image, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(energy_image, energy_image, CV_BGR2GRAY);

	Scharr(energy_image, dx, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	Scharr(energy_image, dy, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);

	convertScaleAbs(dx, abs_dx);
	convertScaleAbs(dy, abs_dy);
	addWeighted(abs_dx, 0.5, abs_dy, 0.5, 0, grad);

	grad.convertTo(energy_image, CV_64F, 1.0 / 255.0);

	//calculate the comulative energy
	double a, b, d;
	for (int r = 1; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			a = energy_image.at<double>(r - 1, max(c - 1, 0));
			b = energy_image.at<double>(r - 1, c);
			d = energy_image.at<double>(r - 1, min(c + 1, cols - 1));

			energy_image.at<double>(r, c) += min(a, min(b, d));
		}
	}

	return energy_image;
}

void removeCol(Mat& image) {
	Mat energy_image = energe(image);
	int rows_minus1 = energy_image.rows - 1, cols_minus1 = energy_image.cols - 1, *path = new int[rows_minus1 + 1];

	//find the smallest pixel in the last row
	path[rows_minus1] = 0;
	for (int c = 1; c <= cols_minus1; ++c) {
		path[rows_minus1] = (energy_image.at<double>(rows_minus1, c) < energy_image.at<double>(rows_minus1, path[rows_minus1]) ? c : path[rows_minus1]);
	}

	//find the smallest pixels in the upper rows
	for (int r = rows_minus1 - 1; r >= 0; --r){
		path[r] = path[r + 1];
		if (path[r + 1] > 0 && energy_image.at<double>(r, path[r + 1] - 1) < energy_image.at<double>(r, path[r]))
			path[r] = path[r] - 1;
		if (path[r + 1] < cols_minus1 && energy_image.at<double>(r, path[r + 1] + 1) < energy_image.at<double>(r, path[r]))
			path[r] = path[r] + 1;
	}

	Mat output(rows_minus1 + 1, cols_minus1, CV_8UC3);
	//remove pixels
	for (int r = 0; r <= rows_minus1; ++r) {
		int c = 0;
		while (c < path[r] - 1) {
			output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c); ++c;
		}

		//average the bounder
		if (c == path[r] - 1) {
			avgPixel(image.at<Vec3b>(r, c), image.at<Vec3b>(r, c + 1), output.at<Vec3b>(r, c)); ++c;
		}
		if (c < cols_minus1) {
			avgPixel(image.at<Vec3b>(r, c), image.at<Vec3b>(r, c + 1), output.at<Vec3b>(r, c));
		}

		for (++c; c < cols_minus1; ++c) {
			output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c + 1);
		}
	}

	image = output;
}

void removeRow(Mat& image) {
	//rotate clockwise 90 degree
	transpose(image, image);
	flip(image, image, 1);

	removeCol(image);

	//rotate counterclockwise 90 degree
	transpose(image, image);
	flip(image, image, 0);
}

void seamCarving(Mat& image) {
	cout << "The original image has " << image.cols << " * " << image.rows << " pixels."<< endl;
	int r = image.rows / 2, c = image.cols / 2;
	for (int i = 0; i < c; ++i) {
		removeCol(image);
		cout << "Progress: " << (i * 100.0 / (r + c)) << "%" << endl;
	}

	for (int i = 0; i < r; ++i) {
		removeRow(image);
		cout << "Progress: " << ((c + i) * 100.0 / (r + c)) << "%" << endl;
	}
	cout << "The processed image has " << image.cols << " * " << image.rows << " pixels." << endl;
}

int main(int argc, char** argv){
	String filename = (argc >= 2) ? argv[1] : "default.jpg";
	
	Mat image;
	image = imread(filename, IMREAD_COLOR); 
	if (image.empty()) {
		cout << "The path is incorrect." << endl;
		return 1;
	}
	seamCarving(image);

	int point_pos = filename.find_last_of(".");
	filename = filename.substr(0, point_pos) + "_compressed" + filename.substr(point_pos);
	imwrite(filename, image);
}