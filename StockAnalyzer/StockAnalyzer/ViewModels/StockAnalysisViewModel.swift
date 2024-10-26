//
//  StockAnalysisViewModel.swift
//  StockAnalyzer
//
//  Created by sunghwan ki on 10/25/24.
//
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

// ViewModels/StockAnalysisViewModel.swift 수정
class StockAnalysisViewModel: ObservableObject {
    @Published var result: StockAnalysisResponse?
    @Published var isLoading = false
    @Published var error: String?
    
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
                
                // 디버깅을 위해 JSON 응답 출력
                if let jsonString = String(data: data, encoding: .utf8) {
                    print("Received JSON response: \(jsonString)")
                }
                
                do {
                    let decoder = JSONDecoder()
                    // decoder.keyDecodingStrategy = .convertFromSnakeCase  // 이 줄 제거
                    let result = try decoder.decode(StockAnalysisResponse.self, from: data)
                    self?.result = result
                } catch {
                    self?.error = "Failed to decode response: \(error.localizedDescription)"
                    print("Decoding error details: \(error)")
                }
            }
        }.resume()
    }
}
