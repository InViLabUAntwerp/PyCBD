#include "libcbdetect/boards_from_corners.h"
#include "libcbdetect/config.h"
#include "libcbdetect/find_corners.h"
#include "libcbdetect/plot_boards.h"
#include "libcbdetect/plot_corners.h"
#include "libcbdetect/Checkerboard.cpp"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>


using namespace std::chrono;
using namespace std;

void detect(const char* str, cbdetect::CornerType corner_type) {
  cbdetect::Corner corners;
  Checkerboard C = Checkerboard();
  std::vector<cbdetect::Board> boards;
  cbdetect::Params params;
  C.norm = 1;
  //params.strict_grow = 0;
  //params.overlay = 0;
  C.score_thr = 0.01;
  //C.corner_type = corner_type;
  C.strict_grow = 0;
  //params.show_processing = 0;
  C.show_grow_processing =0;
  C.overlay = 1;
  C.show_debug_image = 0;

  C.load_image(str);
  C.add_corner(15, 15, 0, -1, 1, 0, 0.15);
  auto t1 = high_resolution_clock::now();
  C.find_corners();  
  C.add_corner_naive(55,55);
  C.add_corner(15, 15, 0, -1, 1, 0, 0.15);
  C.refine_corners();
  auto t2 = high_resolution_clock::now(); 
  C.plot_corners();

  auto t3 = high_resolution_clock::now();
  C.find_board_from_corners();
  auto t4 = high_resolution_clock::now();
  printf("Find corners took: %.3f ms\n", duration_cast<microseconds>(t2 - t1).count() / 1000.0);
  printf("Find boards took: %.3f ms\n", duration_cast<microseconds>(t4 - t3).count() / 1000.0);
  printf("Total took: %.3f ms\n", duration_cast<microseconds>(t2 - t1).count() / 1000.0 + duration_cast<microseconds>(t4 - t3).count() / 1000.0);
  C.plot_boards();
}

int main(int argc, char* argv[]) {

	vector<cv::String> fn;
	cv::glob("../../example_data/Images/Thermal/*.tiff", fn, false);

	vector<cv::Mat> images;
	size_t count = fn.size(); //number of png files in images folder
	std::string str = "";
		for (size_t i = 0; i < count; i++)
		{
			str = fn[i];

			const char* str2 = fn[i].c_str();
		    //detect(str2, cbdetect::SaddlePoint);
		}




	printf("deltilles...");
	detect("../../example_data/IMG_2264_flare.jpg", cbdetect::SaddlePoint);
	printf("deltilles...");
	detect("../../example_data/31-1.tiff", cbdetect::SaddlePoint);
	printf("chessboards...");
	detect("../../example_data/32-1.tiff", cbdetect::SaddlePoint);
	printf("chessboards...");
  detect("../../example_data/image4.jpg", cbdetect::SaddlePoint);

  printf("chessboards...");
  detect("../../example_data/tst2.bmp", cbdetect::SaddlePoint);
  printf("chessboards...");
  detect("../../example_data/tst_5.bmp", cbdetect::SaddlePoint);
  printf("chessboards...");
  detect("../../example_data/image4.jpg", cbdetect::SaddlePoint);
  printf("chessboards...");
  detect("../../example_data/im10.jpg", cbdetect::SaddlePoint);


  printf("deltilles...");
  detect("../../example_data/image16.png", cbdetect::SaddlePoint);
  printf("deltilles...");
  detect("../../example_data/Op3Mechgirl3.png", cbdetect::SaddlePoint);
  printf("deltilles...");
  detect("../../example_data/image.jpg", cbdetect::SaddlePoint);
  printf("deltilles...");
  detect("../../example_data/e3.png", cbdetect::SaddlePoint);
  printf("deltilles...");
  detect("../../example_data/11-1.tiff", cbdetect::SaddlePoint);
  return 0;
}
