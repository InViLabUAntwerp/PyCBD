# pragma once
#include "libcbdetect/boards_from_corners.h"
#include "libcbdetect/config.h"
#include "libcbdetect/find_corners.h"
#include "libcbdetect/find_subpixel_corners.h"
#include "libcbdetect/plot_boards.h"
#include "libcbdetect/plot_corners.h"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>

#ifndef SWIG
using namespace cv;
#endif SWIG
class Checkerboard
{
public:
    int rows;

    int cols;
	int number_of_boards;
	int number_of_corners;
	
	bool show_processing;
	bool show_debug_image;
	bool show_grow_processing;
	bool norm;
	bool polynomial_fit;
	int norm_half_kernel_size;
	int polynomial_fit_half_kernel_size;
	double init_loc_thr;
	double score_thr;
	bool strict_grow;
	bool overlay;
	bool occlusion;

	std::vector<cbdetect::Board> boards;
	std::vector<int> radius;


	cbdetect::Corner corners;
#ifndef SWIG
	cv::Mat image;
#endif SWIG


	Checkerboard()
	{
		this->cols = 0;
		this->rows = 0;
		this->number_of_boards = 0;
		this->number_of_corners = 0;
		this->show_processing=true;
		this->show_debug_image = (false);
		this->show_grow_processing = (false);
		this->norm = (false);
		this->polynomial_fit = (true);
		this->norm_half_kernel_size = (31);
		this->polynomial_fit_half_kernel_size = (4);
		this->init_loc_thr = (0.01);
		this->score_thr = (0.01);
		this->strict_grow = (true);
		this->overlay = (false);
		this->occlusion = (true);
	

	}
	~Checkerboard()
	{
		// deallocate memory 
		std::cout << "DESTRUCTOR 0705" << std::endl;
		corners.v1.clear();
		corners.v2.clear();
		corners.v3.clear();
		corners.score.clear();
		std::cout << "DESTRUCTOR 0705 corner" << std::endl;

		boards.clear();
				std::cout << "DESTRUCTOR 0705 board" << std::endl;

		
	}
	void array_to_image(int sizex, double* arrx, int h, int w)
	{
		int height = h;
		int width = w;
		// Create matrix to store the image
		cv::Mat image_matrix = cv::Mat(height, width, CV_8UC1);

		// Copy array values to matrix
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				image_matrix.at<uchar>(i, j) = arrx[i * width + j];
			}
		}

		// Copy matrix to image
		image_matrix.copyTo(this->image);
	}
	void array_norm_to_image(int sizex,double* arrx, int h, int w)
	{
		// Create matrix to store the image
		int height = h;
		int width = w;
		
		cv::Mat image_matrix = cv::Mat(height,width, CV_64F);  //CV_64F

		// Copy array values to matrix
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				float f;

				f = static_cast<double>(arrx[i * width + j]);
				image_matrix.at<double>(i, j) =f;
				
			}
		}

		// Copy matrix to image
		image_matrix.copyTo(this->image);
	}
	int load_image(std::string name) {
		this->image = cv::imread(name); /* TODO 0705  this gives som problems with rotated images,  check if we can fix this (low prio) */
	
		// Check for invalid input
		if (image.empty())
		{
			std::cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		return 0;
	}
	int find_corners()
	{
		cbdetect::Corner corner;
		this->corners = corner;
		cbdetect::Params params = this->generateparams();
		cbdetect::find_corners(this->image, this->corners, params);
		this->number_of_corners = corners.p.size();
		return 0;
	}
#ifndef SWIG
	cbdetect::Params generateparams()
	{
		cbdetect::Params params;
		params.show_processing = this->show_processing ;
		params.show_debug_image = this->show_debug_image;
		params.show_grow_processing = this->show_grow_processing ;
		params.norm = this->norm  ;
		params.polynomial_fit = this->polynomial_fit ;
		params.norm_half_kernel_size = this->norm_half_kernel_size ;
		params.polynomial_fit= this->polynomial_fit_half_kernel_size;
		params.init_loc_thr = this->init_loc_thr;
		params.score_thr = this->score_thr ;
		params.strict_grow = this->strict_grow ;
		params.overlay = this->overlay ;
		params.occlusion = this->occlusion ;

		return params;
	}
#endif SWIG
	int find_board_from_corners()
	{
		std::vector<cbdetect::Board> boardsnew;
		this->boards = boardsnew;
		cbdetect::boards_from_corners(this->image, this->corners,  this->boards, this->generateparams());
		this->CalculateRowsCols();
		return 1;
	}
	void CalculateRowsCols() {


		this->number_of_corners = corners.p.size();
		for (int n = 0; n < boards.size(); ++n) {

			const auto& board = boards[n];
			this->rows = board.idx.size();
			this->cols = board.idx[1].size();
			this->number_of_boards = n;


			
			
		}
	}
	void GetBoardCorners(int sizex, double* arrx, int sizey, double* arry) {
		std::cout << "copy board corners start " << boards.size() << std::endl;
		for (int n = 0; n < boards.size(); ++n) {

			const auto& board = boards[boards.size()-1];


			for (int i = 0; i < board.idx.size() * board.idx[0].size(); ++i) { // TODO: 0705 this for loop is not needed
				int row = i / board.idx[0].size();
				int col = i % board.idx[0].size();
				cv::Point2d c = corners.p[board.idx[row][col]];
				arrx[i] = c.x;
				arry[i] = c.y;
			}
		}

	}
	void GetCorners(int sizex, double* arrx, int sizey, double* arry){
		for (int i = 0; i < corners.p.size() ; ++i) {

			cv::Point2d c = corners.p[i];
			arrx[i] = c.x;
			arry[i] = c.y;
		}
	}
	void GetDirection_U(int sizex, double* arrx, int sizey, double* arry) {
		for (int i = 0; i < corners.v1.size(); ++i) {

			cv::Point2d c = corners.v1[i];
			arrx[i] = c.x;
			arry[i] = c.y;
		}
	}

	void GetDirection_V(int sizex, double* arrx, int sizey, double* arry) {
		for (int i = 0; i < corners.v2.size(); ++i) {

			cv::Point2d c = corners.v2[i];
			arrx[i] = c.x;
			arry[i] = c.y;
		}
	}

	void GetScore(int sizex, double* arrx) {
		for (int i = 0; i < corners.score.size(); ++i) {

			arrx[i] = corners.score[i];
		
		}
	}

	void add_corner_naive(float u, float v) {

		cv::Point2d point;
		point.x = u;
		point.y = v;
		this->corners.p.emplace_back(point);
		this->corners.r.emplace_back(this->corners.r.back());
		this->corners.v1.emplace_back(this->corners.v1.back());
		this->corners.v2.emplace_back(this->corners.v2.back());
		this->corners.score.emplace_back(this->corners.score.back());
		this->number_of_corners = corners.p.size();
	}

	void add_corner(float u, float v, float direction_U_u, float direction_U_v, float direction_V_u, float direction_V_v, float score) {

		cv::Point2d point;
		point.x = u;
		point.y = v;
		cv::Point2d v1;
		v1.x = direction_U_u;
		v1.y = direction_U_v;
		cv::Point2d v2;
		v2.x = direction_V_u;
		v2.y = direction_V_v;
		this->corners.p.emplace_back(point);
		if (corners.r.empty()) {
			corners.r.push_back(5); //TODO 0705: figure out what this does (search radius for checker?)
		}
		else {
			this->corners.r.emplace_back(this->corners.r.back());
		}
		this->corners.v1.emplace_back(v1);
		this->corners.v2.emplace_back(v2);
		this->corners.score.emplace_back(score);
		this->number_of_corners = corners.p.size();
	}

	void refine_corners()
	{   /* TODO 0705: piramid search (resolution) is broken */
		//corners.r.clear();
		corners.v1.clear();
		corners.v2.clear();
		corners.v3.clear();
		corners.score.clear();	
		cbdetect::Params params = this->generateparams();
		params.polynomial_fit = false;
		cbdetect::find_subpixel_corners(this->image, this->corners, params);
		this->number_of_corners = corners.p.size();

	}

	
	void plot_corners() {

		cbdetect::plot_corners(this->image, this->corners);
	}
	void plot_boards() {
		cbdetect::plot_boards(this->image,this-> corners,this-> boards,this->generateparams());
	}

};
