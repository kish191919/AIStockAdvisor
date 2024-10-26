//
//  MarketIndicatorsSection.swift
//  StockAnalyzer
//
//  Created by sunghwan ki on 10/25/24.
//
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

struct MarketIndicatorsSection: View {
    let result: StockAnalysisResponse
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Market Indicators")
                .font(.headline)
            
            if let vix = result.vixIndex {  // vix_index -> vixIndex
                HStack {
                    Text("VIX Index")
                    Spacer()
                    Text(String(format: "%.2f", vix))
                }
            }
            
            HStack {
                Text("Fear & Greed Index")
                Spacer()
                Text(String(format: "%.1f", result.fearGreedIndex.value))  // Int -> Double로 형식 변경
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
