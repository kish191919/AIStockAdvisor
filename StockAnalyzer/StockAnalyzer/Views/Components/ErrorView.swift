//
//  ErrorView.swift
//  StockAnalyzer
//
//  Created by sunghwan ki on 10/25/24.
//
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

struct ErrorView: View {
    let message: String
    
    var body: some View {
        Text(message)
            .foregroundColor(.red)
            .padding()
    }
}
