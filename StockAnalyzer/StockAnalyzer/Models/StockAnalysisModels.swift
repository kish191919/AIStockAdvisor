//
//  StockAnalysisModels.swift
//  StockAnalyzer
//
//  Created by sunghwan ki on 10/25/24.
//
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

struct StockAnalysisRequest: Encodable {
    let symbol: String
    let language: String
}

struct StockAnalysisResponse: Decodable {
    let decision: String
    let percentage: Int
    let reason: String
    let current_price: Double
    let expected_next_day_price: Double
    let vix_index: Double?
    let fear_greed_index: FearGreedIndex
    let news: NewsData
}

struct FearGreedIndex: Decodable {
    let value: Int
    let description: String
    let last_update: String
}

struct NewsData: Decodable {
    let google_news: [NewsItem]
    let alpha_vantage_news: [NewsItem]
    let robinhood_news: [NewsItem]
}

struct NewsItem: Decodable {
    let title: String
    let date: String?
    let published_at: String?
    let pubDate: String?
    
    var displayDate: String {
        return date ?? published_at ?? pubDate ?? "No date"
    }
}
