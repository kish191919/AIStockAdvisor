import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

// ViewModels/StockAnalysisViewModel.swift 수정
class StockAnalysisViewModel: ObservableObject {
    @Published var result: StockAnalysisResponse?
    @Published var isLoading = false
    @Published var error: String?
    @Published var chartData: ChartData?
    
    func analyzeStock(symbol: String, language: String) {
        isLoading = true
        error = nil
        
        let request = StockAnalysisRequest(symbol: symbol, language: language)
        
        guard let url = URL(string: "https://aistockadvisor.net/api/analyze") else {
            error = "Invalid URL"
            isLoading = false
            return
        }
        
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            urlRequest.httpBody = try JSONEncoder().encode(request)
        } catch {
            self.error = "Failed to encode request: \(error.localizedDescription)"
            isLoading = false
            return
        }
        
        URLSession.shared.dataTask(with: urlRequest) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.isLoading = false
                
                if let error = error {
                    self?.error = "Network error: \(error.localizedDescription)"
                    return
                }
                
                guard let data = data else {
                    self?.error = "No data received"
                    return
                }
                
                // 받은 JSON 데이터 출력
                if let jsonString = String(data: data, encoding: .utf8) {
                    print("Received JSON response: \(jsonString)")
                }
                
                do {
                    let decoder = JSONDecoder()
                    let response = try decoder.decode(StockAnalysisResponse.self, from: data)
                    self?.result = response
                    
                    // 차트 데이터 디버깅
                    if let chartData = response.chartData {
                        print("Chart data received:")
                        print("Daily data count: \(chartData.dailyData.count)")
                        if let firstDaily = chartData.dailyData.first {
                            print("First daily data: \(firstDaily)")
                        }
                        print("Monthly data count: \(chartData.monthlyData.count)")
                        if let firstMonthly = chartData.monthlyData.first {
                            print("First monthly data: \(firstMonthly)")
                        }
                        self?.chartData = chartData
                    } else {
                        print("No chart data in response")
                    }
                } catch {
                    self?.error = "Failed to decode response: \(error.localizedDescription)"
                    print("Decoding error details: \(error)")
                }
            }
        }.resume()
    }
}
