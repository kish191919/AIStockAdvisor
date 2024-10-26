//
//  DecisionSection.swift
//  StockAnalyzer
//
//  Created by sunghwan ki on 10/25/24.
//

import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

// DecisionSection.swift 수정
struct DecisionSection: View {
    let result: StockAnalysisResponse
    
    var percentageText: String {
        if result.decision != "HOLD" {
            return "\(result.percentage)%"
        }
        return ""
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Analysis Result")
                .font(.headline)
            
            HStack {
                Text(result.decision)
                    .font(.title)
                    .foregroundColor(decisionColor)
                
                if !percentageText.isEmpty {
                    Text(percentageText)
                        .font(.title2)
                }
            }
            
            Text(result.reason)
                .font(.body)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
    }
    
    var decisionColor: Color {
        switch result.decision {
        case "BUY": return .green
        case "SELL": return .red
        default: return .orange
        }
    }
}
