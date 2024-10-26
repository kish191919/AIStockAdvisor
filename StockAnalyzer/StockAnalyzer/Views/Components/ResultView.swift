//Views/Components/ResultView.swift
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

struct ResultView: View {
    let result: StockAnalysisResponse
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Decision Section
            DecisionSection(result: result)
            
            // Price Section
            PriceSection(result: result)
            
            // Market Indicators Section
            MarketIndicatorsSection(result: result)
            
            // NewsListView는 여기서 제거
        }
        .padding()
    }
}
