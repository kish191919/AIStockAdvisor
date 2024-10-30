import Foundation

struct SearchResponse: Codable {
    let quotes: [Quote]
}

struct Quote: Codable {
    let symbol: String
}
