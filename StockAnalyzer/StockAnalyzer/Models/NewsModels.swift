// Models/NewsModels.swift
import Foundation

// MARK: - News Models
struct StockNewsData: Decodable {
    let googleNews: [StockNewsItem]
    let alphaVantageNews: [StockNewsItem]
    let robinhoodNews: [StockNewsItem]
    
    enum CodingKeys: String, CodingKey {
        case googleNews = "google_news"
        case alphaVantageNews = "alpha_vantage_news"
        case robinhoodNews = "robinhood_news"
    }
}

struct StockNewsItem: Identifiable, Decodable {
    let id = UUID()
    let title: String
    let date: String?
    let published_at: String?
    let pubDate: String?
    let url: String?
    
    var displayDate: String {
        return date ?? published_at ?? pubDate ?? "No date"
    }
    
    enum CodingKeys: String, CodingKey {
        case title
        case date
        case published_at
        case pubDate
        case url
    }
}
