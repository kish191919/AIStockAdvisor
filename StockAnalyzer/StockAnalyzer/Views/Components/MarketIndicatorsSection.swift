
//Views/Components/MarketIndicatorsSection.swift
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

struct MarketIndicatorsSection: View {
    let result: StockAnalysisResponse
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Market Indicators")
                .font(.headline)
            
            if let vix = result.vixIndex {
                HStack {
                    Text("VIX Index")
                    Spacer()
                    Text(String(format: "%.2f", vix))
                }
            }
            
            HStack {
                Text("Fear & Greed Index")
                Spacer()
                Text(String(format: "%.1f", result.fearGreedIndex.value))
            }
            
            Text(result.fearGreedIndex.description)
                .font(.caption)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
    }
}
