import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

// ViewModels/StockAnalysisViewModel.swift
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
                
                // API 에러 응답 체크
                if let errorResponse = try? JSONDecoder().decode(APIErrorResponse.self, from: data) {
                    if errorResponse.detail.contains("429") {
                        self?.error = "Server is currently busy. Please try again later."
                        return
                    }
                    self?.error = "Server error: \(errorResponse.detail)"
                    return
                }
                
                do {
                    let decoder = JSONDecoder()
                    let response = try decoder.decode(StockAnalysisResponse.self, from: data)
                    self?.result = response
                    self?.chartData = response.chartData
                } catch {
                    self?.error = "Failed to decode response: \(error.localizedDescription)"
                    print("Decoding error details: \(error)")
                }
            }
        }.resume()
    }
}

// API 에러 응답을 위한 새로운 모델
struct APIErrorResponse: Codable {
    let detail: String
}
