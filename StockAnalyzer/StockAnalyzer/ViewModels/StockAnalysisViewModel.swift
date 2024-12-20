
import SwiftUI
import Combine
import CoreData
import Foundation

// ViewModels/StockAnalysisViewModel.swift
@MainActor
class StockAnalysisViewModel: ObservableObject {
    @Published var result: StockAnalysisResponse?
    @Published var isLoading = false
    @Published var error: String?
    
    // analyzeStock 함수
    func analyzeStock(symbol: String, language: String) async {
        isLoading = true
        error = nil
        
        let request = StockAnalysisRequest(symbol: symbol, language: language)
        
        guard let url = URL(string: "https://aistockadvisor.net/api/analyze") else {
//        guard let url = URL(string: "http://localhost:8000/api/analyze") else {
            error = "Invalid URL"
            isLoading = false
            return
        }
        
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            urlRequest.httpBody = try JSONEncoder().encode(request)
            
            let (data, _) = try await URLSession.shared.data(for: urlRequest)
            
            if let errorResponse = try? JSONDecoder().decode(APIErrorResponse.self, from: data) {
                if errorResponse.detail.contains("429") {
                    error = "Server is currently busy. Please try again later."
                    return
                }
                error = "Server error: \(errorResponse.detail)"
                return
            }
            
            let decoder = JSONDecoder()
            result = try decoder.decode(StockAnalysisResponse.self, from: data)
            
        } catch {
            self.error = "Error: \(error.localizedDescription)"
            print("Error details: \(error)")
        }
        
        isLoading = false
    }
    
    // savePrediction 함수 추가
    func savePrediction(symbol: String) {
        guard let result = self.result else { return }
        
        let context = PersistenceController.shared.container.viewContext
        let prediction = NSEntityDescription.insertNewObject(forEntityName: "StockPrediction", into: context) as! NSManagedObject
        
        prediction.setValue(UUID(), forKey: "id")
        prediction.setValue(symbol, forKey: "symbol")
        prediction.setValue(Date(), forKey: "analysisDate")
        prediction.setValue(result.decision, forKey: "decision")
        prediction.setValue(result.currentPrice, forKey: "currentPrice")
        prediction.setValue(result.expectedNextDayPrice, forKey: "expectedNextDayPrice")
        prediction.setValue(result.reason, forKey: "reason")
        prediction.setValue(false, forKey: "isUpdate")
        prediction.setValue(0.0, forKey: "actualClosedPrice")
        prediction.setValue(0.0, forKey: "accuracy")
        
        do {
            try context.save()
            print("Successfully saved prediction for \(symbol)")
        } catch {
            print("Failed to save prediction: \(error)")
        }
    }
}

// API 에러 응답을 위한 새로운 모델
struct APIErrorResponse: Codable {
    let detail: String
}

extension StockAnalysisViewModel {
    @MainActor
    func retryAnalysis(symbol: String, language: String) async {
        // 이전 상태 초기화
        result = nil
        error = nil
        isLoading = true
        
        // 분석 다시 시도
        await analyzeStock(symbol: symbol, language: language)
    }
}
