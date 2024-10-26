//
//  StockAnalysisModels.swift
//  StockAnalyzer
//
//  Created by sunghwan ki on 10/25/24.
//
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

// Models/StockAnalysisModels.swift 수정
struct StockAnalysisRequest: Encodable {
    let symbol: String
    let language: String
    
    enum CodingKeys: String, CodingKey {
        case symbol
        case language
    }
}

struct StockAnalysisResponse: Decodable {
    let decision: String
    let percentage: Int
    let reason: String
    let currentPrice: Double        // current_price -> currentPrice
    let expectedNextDayPrice: Double // expected_next_day_price -> expectedNextDayPrice
    let vixIndex: Double?          // vix_index -> vixIndex
    let fearGreedIndex: FearGreedIndex  // fear_greed_index -> fearGreedIndex
    let news: NewsData
    
    enum CodingKeys: String, CodingKey {
        case decision
        case percentage
        case reason
        case currentPrice = "current_price"
        case expectedNextDayPrice = "expected_next_day_price"
        case vixIndex = "vix_index"
        case fearGreedIndex = "fear_greed_index"
        case news
    }
}

struct FearGreedIndex: Decodable {
    let value: Double  // Int -> Double로 변경 (JSON에서 58.9142857142857와 같은 소수점 값이 오고 있음)
    let description: String
    let lastUpdate: String
    
    enum CodingKeys: String, CodingKey {
        case value
        case description
        case lastUpdate = "last_update"
    }
}

struct NewsData: Decodable {
    let googleNews: [NewsItem]      // google_news -> googleNews
    let alphaVantageNews: [NewsItem] // alpha_vantage_news -> alphaVantageNews
    let robinhoodNews: [NewsItem]    // robinhood_news -> robinhoodNews
    
    enum CodingKeys: String, CodingKey {
        case googleNews = "google_news"
        case alphaVantageNews = "alpha_vantage_news"
        case robinhoodNews = "robinhood_news"
    }
}

struct NewsItem: Decodable {
    let title: String
    let date: String?
    let published_at: String?
    let pubDate: String?
    
    // CodingKeys 추가
    enum CodingKeys: String, CodingKey {
        case title
        case date
        case published_at = "published_at"
        case pubDate
    }
    
    var displayDate: String {
        return date ?? published_at ?? pubDate ?? "No date"
    }
}
