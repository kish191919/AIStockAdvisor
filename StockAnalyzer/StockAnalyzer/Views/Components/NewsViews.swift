//Views/Components/NewsViews.swift
import SwiftUI
import SafariServices

// MARK: - Main News List View
struct NewsListView: View {
    let news: StockNewsData
    @State private var selectedURL: URL?
    @State private var showingSafari = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Latest News")
                .font(.title2)
                .padding(.horizontal)
            
            if !news.yahooFinanceNews.isEmpty {
                NewsSourceSection(
                    title: "Yahoo Finance News",
                    items: news.yahooFinanceNews,
                    selectedURL: $selectedURL,
                    showingSafari: $showingSafari
                )
            }
            
            if !news.alphaVantageNews.isEmpty {
                NewsSourceSection(
                    title: "Alpha Vantage News",
                    items: news.alphaVantageNews,
                    selectedURL: $selectedURL,
                    showingSafari: $showingSafari
                )
            }
            
            if !news.robinhoodNews.isEmpty {
                NewsSourceSection(
                    title: "Robinhood News",
                    items: news.robinhoodNews,
                    selectedURL: $selectedURL,
                    showingSafari: $showingSafari
                )
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
        .sheet(isPresented: $showingSafari) {
            if let url = selectedURL {
                SafariView(url: url)
            }
        }
    }
}

// MARK: - News Source Section
struct NewsSourceSection: View {
    let title: String
    let items: [StockNewsItem]
    @Binding var selectedURL: URL?
    @Binding var showingSafari: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
                .foregroundColor(.secondary)
                .padding(.horizontal)
            
            ForEach(items.prefix(3)) { item in
                NewsItemRow(item: item) {
                    if let urlString = item.url,
                       let url = URL(string: urlString) {
                        selectedURL = url
                        showingSafari = true
                    }
                }
                
                if item.id != items.last?.id {
                    Divider()
                }
            }
        }
        .padding(.vertical, 8)
    }
}

// MARK: - News Item Row
struct NewsItemRow: View {
    let item: StockNewsItem
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 4) {
                Text(item.title)
                    .foregroundColor(item.url != nil ? .blue : .primary)
                    .multilineTextAlignment(.leading)
                    .lineLimit(2)
                Text(item.displayDate)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal)
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(PlainButtonStyle())
        .disabled(item.url == nil || URL(string: item.url ?? "") == nil)
    }
}

// MARK: - Safari View
struct SafariView: UIViewControllerRepresentable {
    let url: URL
    
    func makeUIViewController(context: Context) -> SFSafariViewController {
        return SFSafariViewController(url: url)
    }
    
    func updateUIViewController(_ uiViewController: SFSafariViewController, context: Context) {}
}

// MARK: - Preview Provider
struct NewsViews_Previews: PreviewProvider {
    static var previews: some View {
        NewsListView(news: StockNewsData(
            yahooFinanceNews: [
                StockNewsItem(
                    title: "Sample Yahoo Finance News",
                    date: "2024-01-01",
                    published_at: nil,
                    pubDate: nil,
                    url: "https://example.com"
                )
            ],
            alphaVantageNews: [
                StockNewsItem(
                    title: "Sample Alpha Vantage News",
                    date: nil,
                    published_at: "2024-01-01",
                    pubDate: nil,
                    url: "https://example.com"
                )
            ],
            robinhoodNews: [
                StockNewsItem(
                    title: "Sample Robinhood News",
                    date: nil,
                    published_at: nil,
                    pubDate: "2024-01-01",
                    url: "https://example.com"
                )
            ]
        ))
        .padding()
    }
}
