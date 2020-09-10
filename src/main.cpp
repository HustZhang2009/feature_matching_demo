#include "feature_matching.hpp"
#include <iostream>
#include <memory>

int main(int argc,char** argv)
{
    std::cout << "feature match demo" << std::endl;
    if(argc<2)
    {
        std::cout << "usge : need one param to set the config path" << std::endl;
        return 0;
    }

    std::shared_ptr<FeatureMatching> match_demo = std::make_shared<FeatureMatching>();
    match_demo->set_config(std::string(argv[1]));
    match_demo->run();

    return 0;
}