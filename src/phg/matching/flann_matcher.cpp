#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(2); // видимо, количество рандомных деревьев
    search_params = flannKsTreeSearchParams(60); // видимо, количество ступеней наверх до остановки алгоритма
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat indices;
    cv::Mat dists;
    flann_index->knnSearch(query_desc, indices, dists, k, *search_params);
    for(int i = 0; i < indices.size[0]; ++i)
    {
        std::vector<cv::DMatch> v;
        for(int j = 0; j < k; ++j)
        {
            cv::DMatch match(i, indices.at<int>(i, j), dists.at<float>(i, j));
            v.push_back(match);
        }
        matches.push_back(v);
    }
}
