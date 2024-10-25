//
//  PriceSection.swift
//  StockAnalyzer
//
//  Created by sunghwan ki on 10/25/24.
//
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

struct PriceSection: View {
    let result: StockAnalysisResponse
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Price Information")
                .font(.headline)
            
            HStack {
                VStack(alignment: .leading) {
                    Text("Current Price")
                        .font(.subheadline)
                    Text("$\(String(format: "%.2f", result.current_price))")
                        .font(.title2)
                }
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text("Expected Next Day")
                        .font(.subheadline)
                    Text("$\(String(format: "%.2f", result.expected_next_day_price))")
                        .font(.title2)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
    }
}
