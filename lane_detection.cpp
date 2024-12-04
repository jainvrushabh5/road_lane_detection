#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

cv::Mat roi(const cv::Mat& image, const std::vector<cv::Point>& vertices) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    
    
    const cv::Point* points[1] = { &vertices[0] };
    int points_size[] = { static_cast<int>(vertices.size()) };
    
    cv::fillPoly(mask, points, points_size, 1, cv::Scalar(255));
    
    
    cv::Mat cropped_img;
    cv::bitwise_and(image, mask, cropped_img);
    
    return cropped_img;
}

cv::Mat draw_lines(cv::Mat image, const std::vector<cv::Vec4i>& hough_lines) {
    for (const auto& line : hough_lines) {
        cv::line(image, 
                 cv::Point(line[0], line[1]), 
                 cv::Point(line[2], line[3]), 
                 cv::Scalar(0, 255, 0), 
                 2);
    }
    return image;
}

cv::Mat process(cv::Mat img) {
    int height = img.rows;
    int width = img.cols;
    

    std::vector<cv::Point> roi_vertices = {
        cv::Point(0, 650),
        cv::Point(2 * width / 3, 2 * height / 3),
        cv::Point(width, 1000)
    };

    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    
 
    cv::Mat dilated;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(gray_img, dilated, kernel);
    
 
    cv::Mat canny;
    cv::Canny(dilated, canny, 130, 220);
    
   
    cv::Mat roi_img = roi(canny, roi_vertices);
    
 
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(roi_img, lines, 1, CV_PI/180, 10, 15, 2);
    
   
    cv::Mat final_img = draw_lines(img, lines);
    
    return final_img;
}

int main() {
 
    cv::VideoCapture cap("./lane_vid2.mp4");
    
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }
    
    
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    
    cv::VideoWriter saved_frame("lane_detection_1.avi", 
                                cv::VideoWriter::fourcc('X','V','I','D'), 
                                30.0, 
                                cv::Size(frame_width, frame_height));
    
    if (!saved_frame.isOpened()) {
        std::cerr << "Error creating video writer" << std::endl;
        return -1;
    }
    
    
    cv::Mat frame;
    while (true) {
        cap >> frame;
        
      
        if (frame.empty()) {
            break;
        }
        
        try {
           
            frame = process(frame);
            
           
            saved_frame.write(frame);
            cv::imshow("Lane Detection", frame);
           
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV Exception: " << e.what() << std::endl;
            break;
        }
    }
    
   
    cap.release();
    saved_frame.release();
    cv::destroyAllWindows();
    
    return 0;
}