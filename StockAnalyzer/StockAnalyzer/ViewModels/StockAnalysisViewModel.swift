//
//  StockAnalysisViewModel.swift
//  StockAnalyzer
//
//  Created by sunghwan ki on 10/25/24.
//
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

class StockAnalysisViewModel: ObservableObject {
    @Published var result: StockAnalysisResponse?
    @Published var isLoading = false
    @Published var error: String?
    
    func analyzeStock(symbol: String, language: String) {
        isLoading = true
        error = nil
        
        let request = StockAnalysisRequest(symbol: symbol, language: language)
        
        guard let url = URL(string: "http://3.210.156.34:8000/api/analyze") else {
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
            self.error = "Failed to encode request"
            isLoading = false
            return
        }
        
        URLSession.shared.dataTask(with: urlRequest) { data, response, error in
            DispatchQueue.main.async {
                self.isLoading = false
                
                if let error = error {
                    self.error = error.localizedDescription
                    return
                }
                
                guard let data = data else {
                    self.error = "No data received"
                    return
                }
                
                do {
                    let result = try JSONDecoder().decode(StockAnalysisResponse.self, from: data)
                    self.result = result
                } catch {
                    self.error = "Failed to decode response: \(error.localizedDescription)"
                }
            }
        }.resume()
    }
}
