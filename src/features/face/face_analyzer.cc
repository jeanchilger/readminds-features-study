#include "src/features/face/face_analyzer.h"

#include <vector>

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"


FaceAnalyzer::FaceAnalyzer(int img_width, int img_height)
    : GenericAnalyzer(img_width, img_height) {}

FaceAnalyzer::FaceAnalyzer(mediapipe::NormalizedLandmarkList landmarks, 
              int img_width, int img_height)
                    : GenericAnalyzer(landmarks, img_width, img_height) {}

double FaceAnalyzer::GetFaceArea() {
    return face_area_;
}

double FaceAnalyzer::GetFaceMotion() {
    return face_motion_;
}

double FaceAnalyzer::GetFaceCOM() {
    return face_com_;
}

// TODO: This may be a ineffective approach.
// Selecting only the outer landmarks could be faster.
void FaceAnalyzer::CalculateFaceArea() {
    std::vector<cv::Point> all_points;
    std::vector<cv::Point> hull;

    int x, y;
    for (int i=0; i < NTOTAL_LANDMARKS; i ++) {
        mediapipe::NormalizedLandmark landmark = landmarks_.landmark(i);
        
        all_points.push_back(CvtNormIntoCvPoint(landmark));
    }

    cv::convexHull(all_points, hull);

    face_area_ = cv::contourArea(hull) / norm_factor_;
}

void FaceAnalyzer::CalculateFaceMotion() {
    double face_motion = 0;
    int anchor_list_size = sizeof(ANCHOR_LANDMARKS) / sizeof(*ANCHOR_LANDMARKS);

    // Equivalent to D(f - Z, )
    mediapipe::NormalizedLandmarkList first_anchor_list = last_anchors_.at(0);

    for (int f=1; f < num_frames_motion_; f++) {
        // Equivalent to D(f - t, )
        mediapipe::NormalizedLandmarkList anchor_list = last_anchors_.at(f);
        
        for (int j=0; j < anchor_list_size; j++) {
            // Equivalent to D(f - t, j)
            mediapipe::NormalizedLandmark anchor_landmark = 
                    anchor_list.landmark(j);

            // Equivalent to D(f - Z, j)
            mediapipe::NormalizedLandmark first_anchor_landmark = 
                    first_anchor_list.landmark(j);

            face_motion += EuclideanDistance(
                    CvtNormIntoCvPoint(anchor_landmark), 
                    CvtNormIntoCvPoint(first_anchor_landmark));
        }
    }

    face_motion_ = face_motion / norm_factor_;
}

void FaceAnalyzer::UpdateLastAnchors() {
    mediapipe::NormalizedLandmarkList anchor_list;
    for (int a : ANCHOR_LANDMARKS) {
        
        mediapipe::NormalizedLandmark* anchor_landmark = 
                anchor_list.add_landmark();

        *anchor_landmark = landmarks_.landmark(a);
    }

    last_anchors_.push_back(anchor_list);

    if (last_anchors_.size() > num_frames_motion_) {
        last_anchors_.pop_front();
    }
}

void FaceAnalyzer::CalculateFacialCenterOfMass() {
    face_com_ = 0;
    for (int i=0; i < NTOTAL_LANDMARKS; i++) {
        mediapipe::NormalizedLandmark lmark = landmarks_.landmark(i);
        face_com_ += EuclideanNorm(CvtNormIntoCvPoint(lmark));
    }
    
    face_com_ = face_com_ / NTOTAL_LANDMARKS / norm_factor_;
}

void FaceAnalyzer::Update() {
    UpdateLastAnchors();
    CalculateFaceArea();
    CalculateFacialCenterOfMass();

    if (last_anchors_.size() == num_frames_motion_) {
        CalculateFaceMotion();
    }
}