#include "opencv2/opencv.hpp"
#include <iostream>
#include "main.h"

cv::RNG rng(12345);

// convertire imaginea in grayscale
void LicensePlate::grayscale(cv::Mat& frame)
{
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
}
// functie pentru a compara ariile a doua contururi
bool compara(std::vector<cv::Point>& contur1, std::vector<cv::Point>& contur2)
{
    const double i = fabs(contourArea(cv::Mat(contur1)));
    const double j = fabs(contourArea(cv::Mat(contur2)));
    return (i < j);
}

// deseneaza conturul placutei pe imagine
void LicensePlate::drawLicensePlate(cv::Mat& img, std::vector<std::vector<cv::Point>>& dreptunghiuri)
{
    const int w = img.cols;
    const int h = img.rows;
    const float ratio_w = w / static_cast<float>(512);
    const float ratio_h = h / static_cast<float>(512);
    std::vector<cv::Rect> nonn_drept;

    // parcurge toti candidatii posibili
    for (std::vector<cv::Point> k : dreptunghiuri)
    {
        cv::Rect drept_bound = cv::boundingRect(k);
        float aspect_ratio = drept_bound.width / static_cast<float>(drept_bound.height);

        // verif aspect ratio și dimensiunile pentru a fi considerat ca si candidat bun
        if (aspect_ratio >= 1 && aspect_ratio <= 6 &&
            drept_bound.width < 0.5 * (float)img.cols && drept_bound.height < 0.5 * (float)img.rows)
        {
            float dif1 = drept_bound.area() - contourArea(k);

            // verif dif dintre aria dreptunghiului si aria conturului
            if (dif1 < 3000) 
            {
                nonn_drept.push_back(drept_bound);
            }
        }
    }

    // convertire dreptunghiuri si filtrarea celor care nu sunt dreptunghiuri
    std::vector<cv::Rect> dreptt;
    for (std::vector<cv::Point> m : dreptunghiuri)
    {
        cv::Rect temp = cv::boundingRect(m);
        float dif = temp.area() - cv::contourArea(m);
        if (dif < 3000)
        {
            dreptt.push_back(temp);
        }
    }

    dreptt.erase(std::remove_if(dreptt.begin(), dreptt.end(), [](cv::Rect temp)
        {
            const float aspect_ratio = temp.width / static_cast<float>(temp.height);
            return aspect_ratio < 1 || aspect_ratio > 6;
        }), dreptt.end());

    // gasirea dreptunghiurilor care se suprapun si desenarea acestora revenind la dimensiunea originala
    for (int i = 0; i < dreptt.size(); i++)
    {
        bool inter = false;
        for (int j = i + 1; j < dreptt.size(); j++)
        {
            if (i == j)
            {
                break;
            }
            inter = ((dreptt[i] & dreptt[j]).area() > 0);
            if (inter)
            {
                break;
            }
        }
        if (!inter)
        {
            cv::Scalar culoare = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            cv::rectangle(img, cv::Point(dreptt[i].x * ratio_w, dreptt[i].y * ratio_h), cv::Point((dreptt[i].x + dreptt[i].width) * ratio_w, (dreptt[i].y + dreptt[i].height) * ratio_h), culoare, 3, cv::LINE_8, 0);
        }
    }

    // deseneaza toate dreptunghiurile care corespund placutelor
    for (cv::Rect dreptunghi : dreptt)
    {
        cv::Scalar culoare = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::rectangle(img, cv::Point(dreptunghi.x * ratio_w, dreptunghi.y * ratio_h), cv::Point((dreptunghi.x + dreptunghi.width) * ratio_w, (dreptunghi.y + dreptunghi.height) * ratio_h), culoare, 3, cv::LINE_8, 0);
    }
}

// gaseste candidatii pentru placute in imagine
std::vector<std::vector<cv::Point>> LicensePlate::locateCandidates(cv::Mat& img)
{
    // reducere dim imagine
    cv::Mat rF = img;
    cv::resize(img, rF, cv::Size(512, 512));

    // convertire imagine in grayscale, dacă este necesar
    if (img.channels() == 3)
    {
        LicensePlate::grayscale(rF);
    }

    // aplicarea morfologiei Black-Hat pentru a evidentia regiunile negre pe fundal alb
    cv::Mat bF;
    cv::Mat Kernel_dreptunghi = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 5));
    cv::morphologyEx(rF, bF, cv::MORPH_BLACKHAT, Kernel_dreptunghi);

    // gasirea regiunilor luminoase bazate pe proprietatea de alb
    cv::Mat lF;
    cv::Mat Kernel_patrat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(rF, lF, cv::MORPH_CLOSE, Kernel_patrat);
    cv::threshold(lF, lF, 0, 255, cv::THRESH_OTSU);

    // aplicarea operatorului Sobel pe imaginea de intensitate a regiunilor luminoase
    cv::Mat X;
    double minVal, maxVal;
    int dx = 1, dy = 0, ddepth = CV_32F, ksize = -1;
    cv::Sobel(bF, X, ddepth, dx, dy, ksize);
    X = cv::abs(X);
    cv::minMaxLoc(X, &minVal, &maxVal);
    X = 255 * ((X - minVal) / (maxVal - minVal));
    X.convertTo(X, CV_8U);

    // aplicarea unui blur si a operatiei close pentru a elimina zgomotul
    cv::GaussianBlur(X, X, cv::Size(5, 5), 0);
    cv::morphologyEx(X, X, cv::MORPH_CLOSE, Kernel_dreptunghi);
    cv::threshold(X, X, 0, 255, cv::THRESH_OTSU);

    // erodarea si dilatarea imaginii rezultate
    cv::erode(X, X, 2);
    cv::dilate(X, X, 2);

    // aplicarea operatorului AND intre rez threshold si regiunile luminate
    cv::bitwise_and(X, X, lF);
    cv::dilate(X, X, 2);
    cv::erode(X, X, 1);

    // gasirea contururilor in imaginea threshold si sortarea lor in functie de marime
    std::vector<std::vector<cv::Point>> contur;
    cv::findContours(X, contur, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contur.begin(), contur.end(), compara);
    std::vector<std::vector<cv::Point>> top_contur;
    top_contur.assign(contur.end() - 5, contur.end()); // contururile cele mai mari in ordine descrescatoare

    return top_contur;
}

// afisarea imagine
void LicensePlate::viewer(const cv::Mat& img, std::string titlu)
{
    cv::imshow(titlu, img);
}

int main(int argc, char** argv)
{
    // initializare
    LicensePlate lp;
    //std::string fn = "img1.jpg";
    //std::string fn = "img2.jpg";
    //std::string fn = "img3.png";
    std::string fn = "img4.jpeg";
    //std::string fn = "img5.jpg";
    //std::string fn = "img6.png";
    //std::string fn = "img7.jpg";
    //std::string fn = "img8.jpeg";
    //std::string fn = "img9.jpg";


    cv::Mat img;
    img = cv::imread(fn, cv::IMREAD_COLOR);
    if (!img.data)
    {
        std::cout << "Imaginea nu a fost găsită sau nu s-a putut deschide" << std::endl;
        return -1;
    }
    std::vector<std::vector<cv::Point>> candidati = lp.locateCandidates(img);
    lp.drawLicensePlate(img, candidati);
    lp.viewer(img, "Rezultat");
    cv::waitKey(0);

    return 0;
}
