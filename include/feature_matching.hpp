#ifndef FEATURE_MATCHING_H
#define FEATURE_MATCHING_H

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

class FeatureMatching
{
public:
    FeatureMatching(){}
    FeatureMatching(const std::string& config_name) { set_config(config_name); }
    void set_config(const std::string& config_name);
    void run();
private:
    std::vector<std::string> desc_types_; // 描述子的类型
    std::vector<std::string> match_method_; // 匹配方法的类型
    std::vector<std::string> file_names_; //  待测试图像的名称
};

void FeatureMatching::set_config(const std::string& file_name)   
{
    std::cout << "load config path : " << file_name << std::endl;

    // clear
    desc_types_.clear();
    match_method_.clear();
    file_names_.clear();

    // load config from .yaml
    cv::FileStorage fs(file_name,cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        std::cout << "load config file : " << file_name << "failed" << std::endl;
        return;
    }

    // load desc_types_
    {
        cv::FileNode fn = fs["descriptors"];
        for(cv::FileNodeIterator it = fn.begin();it!=fn.end();it++)
        {
            std::string dsc_name_temp = (std::string)(*it);
            desc_types_.push_back(dsc_name_temp);
        }
    }

    // load match_method_
    {
        cv::FileNode fn = fs["mathes"];
        for(cv::FileNodeIterator it = fn.begin();it!=fn.end();it++)
        {
            match_method_.push_back((std::string)(*it));
        }
    }

    // load file_names_
    {
        cv::FileNode fn = fs["images"];
        for(cv::FileNodeIterator it = fn.begin();it!=fn.end();it++)
        {
            file_names_.push_back((std::string)(*it));
        }
    }

    // for debug
    std::cout << "================================" << std::endl;
    std::cout << "descriptors : " << "\t";
    for(auto &e : desc_types_){std::cout << e << "\t";}std::cout <<std::endl;
    std::cout << "matches : " << "\t";
    for(auto &e : match_method_){std::cout << e << "\t";}std::cout <<std::endl;
    std::cout << "file : " << "\t";
    for(auto &e : file_names_){std::cout << e << "\t";}std::cout <<std::endl;
}

void FeatureMatching::run()
{
    std::cout << "run demo" << std::endl;
    // load images now only support two images
    cv::Mat image1,image2;
    {
        if(file_names_.size() < 2)
        {
            std::cout << "at least need two images ......" << std::endl;
            return;
        }
        image1 = cv::imread(file_names_[0],cv::IMREAD_GRAYSCALE);
        image2 = cv::imread(file_names_[1],cv::IMREAD_GRAYSCALE);
        if(image1.empty() || image2.empty())
        {
            std::cout << "image1 or image2 empty ......" << std::endl;
            return;
        }else
        {
            std::cout << "laod image sucess" << std::endl;
        }
    }

    cv::Ptr<cv::Feature2D> f2d;
    for(int i=0;i<desc_types_.size();i++)
    {
        // get descriptor
        if(desc_types_[i].compare("AKAZE-DESCRIPTOR_KAZE_UPRIGHT"))
        {
            f2d = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_KAZE_UPRIGHT);
        }
        if(desc_types_[i].compare("AKAZE"))
        {
            f2d = cv::AKAZE::create();
        }
        if(desc_types_[i].compare("ORB"))
        {
            f2d = cv::ORB::create();
        }
        if(desc_types_[i].compare("BRISK"))
        {
            f2d = cv::BRISK::create();
        }

        std::cout << "create feature type : " << desc_types_[i] << std::endl;

        // if empty use default ORB feature
        if(!f2d)
        {
            f2d = cv::ORB::create();
            std::cout << "descriptor empty,use default orb feature" << std::endl;
        }

        // detect keypoints
        std::vector<cv::KeyPoint> key_points1,key_points2;
        cv::Mat dsc1, dsc2;
        try
        {
            f2d->detectAndCompute(image1,cv::Mat(),key_points1,dsc1);
            f2d->detectAndCompute(image2,cv::Mat(),key_points2,dsc2);
        }
        catch(const cv::Exception& e)
        {
            std::cout << e.msg << std::endl;
            continue;
        }


        // match
        std::vector<cv::DMatch> matches; 
        try
        {
            cv::Ptr<cv::DescriptorMatcher> pmatcher;
            for(int j=0;j<match_method_.size();j++)
            {
                std::cout << i << "_" << j << std::endl;
                pmatcher = cv::DescriptorMatcher::create(match_method_[j]);
                pmatcher->match(dsc1,dsc2,matches,cv::Mat());

                // sort the match index by distance
                cv::Mat index;
                int match_size = matches.size();
                cv::Mat tab(match_size,1,CV_32F);
                for(int k=0;k<matches.size();k++)
                {
                    tab.at<float>(k,0) = matches[k].distance;
                }
                cv::sortIdx(tab,index,cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);

                // get the best matches
                std::vector<cv::DMatch> best_matches(0);
                std::vector<cv::KeyPoint> best_keypoints1,best_keypoints2;
                for(int l=0; (l<50)&&(l<matches.size()); l++)
                {
                    best_matches.push_back(matches.at(index.at<int>(l,0)));
                    best_keypoints1.push_back(key_points1.at(best_matches[l].trainIdx));
                    best_keypoints2.push_back(key_points2.at(best_matches[l].queryIdx));
                }
                std::cout << "matches.size() : " << matches.size() << "\t";
                std::cout << "best_matches.size() : " << best_matches.size() << std::endl;

                // show result
                cv::Mat result;
                cv::drawMatches(image1,key_points1,image2,key_points2,best_matches,result);
                //cv::drawMatches(image1,best_keypoints1,image2,best_keypoints2,best_matches,result);
                cv::namedWindow(desc_types_[i]+"_"+match_method_[j],0);
                cv::imshow(desc_types_[i]+"_"+match_method_[j],result);
                cv::waitKey(0);
            }
        }
        catch(const cv::Exception& e)
        {
            std::cout << e.msg << std::endl;
            continue;
        }
    }
}

#endif
