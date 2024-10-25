//
//  NewsSection.swift
//  StockAnalyzer
//
//  Created by sunghwan ki on 10/25/24.
//
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

struct NewsSection: View {
    let news: NewsData
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Latest News")
                .font(.headline)
            
            ForEach(news.google_news.prefix(3), id: \.title) { item in
                NewsItemView(item: item)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
    }
}
