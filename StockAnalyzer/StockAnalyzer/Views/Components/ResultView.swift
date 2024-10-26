
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

// Views/Components/ResultView.swift
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
            
            // News Section
            NewsListView(news: result.news)
        }
        .padding()
    }
}
