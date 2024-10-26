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
            DispatchQueue.main.async {
                self.error = "Invalid URL"
                self.isLoading = false
            }
            return
        }
        
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            urlRequest.httpBody = try JSONEncoder().encode(request)
        } catch {
            DispatchQueue.main.async {
                self.error = "Failed to encode request: \(error.localizedDescription)"
                self.isLoading = false
            }
            return
        }
        
        let task = URLSession.shared.dataTask(with: urlRequest) { [weak self] data, response, error in
            guard let self = self else { return }
            
            DispatchQueue.main.async {
                self.isLoading = false
                
                if let error = error {
                    self.error = "Network error: \(error.localizedDescription)"
                    return
                }
                
                guard let data = data else {
                    self.error = "No data received"
                    return
                }
                
                // Print response for debugging
                if let jsonString = String(data: data, encoding: .utf8) {
                    print("Received JSON response: \(jsonString)")
                }
                
                do {
                    let decoder = JSONDecoder()
                    let response = try decoder.decode(StockAnalysisResponse.self, from: data)
                    self.result = response
                    if let chartData = response.chartData {
                        self.chartData = chartData
                    }
                } catch {
                    self.error = "Failed to decode response: \(error.localizedDescription)"
                    print("Decoding error details: \(error)")
                }
            }
        }
        
        task.resume()
    }
}
