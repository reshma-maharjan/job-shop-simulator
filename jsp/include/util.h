#pragma once

#include <string>
#include <functional>
#include <iostream>
#include <iomanip>
#include <curl/curl.h>

class CurlUtility {
private:
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
        size_t totalSize = size * nmemb;
        output->append((char*)contents, totalSize);
        return totalSize;
    }

    static int ProgressCallback(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
        if (dltotal <= 0) return 0;

        ProgressData* progress = static_cast<ProgressData*>(clientp);
        double fraction = static_cast<double>(dlnow) / static_cast<double>(dltotal);
        int percent = static_cast<int>(fraction * 100.0);

        if (percent > progress->lastPercent) {
            progress->lastPercent = percent;
            std::cout << "\rDownloading: [";
            int pos = 50 * fraction;
            for (int i = 0; i < 50; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << std::setw(3) << percent << "%" << std::flush;
        }

        return 0;
    }

    struct ProgressData {
        int lastPercent = -1;
    };

public:
    static std::string FetchUrl(const std::string& url, bool showProgress = false) {
        CURL* curl = curl_easy_init();
        std::string response;

        if (curl) {
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

            ProgressData progress;
            if (showProgress) {
                curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
                curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, ProgressCallback);
                curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress);
            }

            CURLcode res = curl_easy_perform(curl);
            curl_easy_cleanup(curl);

            if (res != CURLE_OK) {
                throw std::runtime_error("CURL error: " + std::string(curl_easy_strerror(res)));
            }

            if (showProgress) {
                std::cout << std::endl;  // Move to the next line after progress bar
            }
        }

        return response;
    }

    static bool DownloadFile(const std::string& url, const std::string& outputPath, bool showProgress = true) {
        CURL* curl = curl_easy_init();
        if (!curl) {
            return false;
        }

        FILE* fp = fopen(outputPath.c_str(), "wb");
        if (!fp) {
            curl_easy_cleanup(curl);
            return false;
        }

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

        ProgressData progress;
        if (showProgress) {
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, ProgressCallback);
            curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress);
        }

        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        fclose(fp);

        if (showProgress) {
            std::cout << std::endl;  // Move to the next line after progress bar
        }

        return (res == CURLE_OK);
    }
};