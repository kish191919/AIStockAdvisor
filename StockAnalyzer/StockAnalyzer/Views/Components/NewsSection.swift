//
//  NewsSection.swift
//  StockAnalyzer
//
//  Created by sunghwan ki on 10/25/24.
//
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

// Views/Components/NewsSection.swift 수정
struct NewsSection: View {
    let news: NewsData
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Latest News")
                .font(.headline)
            
            ForEach(Array(news.googleNews.prefix(3)), id: \.title) { item in
                NewsItemView(item: item)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
    }
}
