import SwiftUI
import SafariServices
import UIKit

// MARK: - Main News List View
struct NewsListView: View {
    let news: StockNewsData
    @State private var currentNewsIndex: Int = 0
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Latest News")
                    .font(.title2)
                    .foregroundColor(ColorTheme.text)
                
                Spacer()
                
                // 페이지 인디케이터
                HStack(spacing: 4) {
                    ForEach(0..<min(combinedNews.count, 5), id: \.self) { index in
                        Circle()
                            .fill(index == currentNewsIndex ? ColorTheme.accent : ColorTheme.secondaryText)
                            .frame(width: 6, height: 6)
                    }
                }
            }
            .padding(.horizontal)
            
            TabView(selection: $currentNewsIndex) {
                ForEach(Array(combinedNews.enumerated()), id: \.element.id) { index, item in
                    NewsCard(item: item) {
                        if let urlString = item.url,
                           let url = URL(string: urlString) {
                            openInBrowser(url)
                        }
                    }
                    .tag(index)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .frame(height: 140)
        }
        .padding()
        .background(ColorTheme.background)
    }
    
    private var combinedNews: [StockNewsItem] {
        let combined = (news.yahooFinanceNews + news.alphaVantageNews)
            .prefix(10)
        return Array(combined)
    }
    
    private func openInBrowser(_ url: URL) {
        if UIApplication.shared.canOpenURL(url) {
            UIApplication.shared.open(url, options: [:]) { success in
                if !success {
                    print("Failed to open URL")
                }
            }
        }
    }
}

// MARK: - News Card View
struct NewsCard: View {
    let item: StockNewsItem
    let onTap: () -> Void
    @State private var isPressed = false
    
    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                Text(item.title)
                    .font(.headline)
                    .lineLimit(3)
                    .foregroundColor(item.url != nil ? ColorTheme.accent : ColorTheme.text)
                    .multilineTextAlignment(.leading)
                
                Spacer(minLength: 4)
                
                HStack {
                    Text(item.displayDate)
                        .font(.caption)
                        .foregroundColor(ColorTheme.secondaryText)
                    
                    Spacer()
                    
                    if item.url != nil {
                        HStack(spacing: 4) {
                            Text("Read More")
                                .font(.caption)
                            Image(systemName: "arrow.right")
                        }
                        .foregroundColor(ColorTheme.accent)
                    }
                }
            }
            .padding(.vertical, 8)
            .padding(.horizontal, 12)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(ColorTheme.secondaryBackground)
            )
            .scaleEffect(isPressed ? 0.98 : 1.0)
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isPressed)
            .padding(.horizontal)
        }
        .buttonStyle(PlainButtonStyle())
        .disabled(item.url == nil)
        .onLongPressGesture(minimumDuration: .infinity, maximumDistance: .infinity,
            pressing: { pressing in
                withAnimation(.easeInOut(duration: 0.2)) {
                    isPressed = pressing
                }
            },
            perform: { }
        )
    }
}

// MARK: - Preview Provider
struct NewsViews_Previews: PreviewProvider {
    static var previews: some View {
        NewsListView(news: StockNewsData(
            yahooFinanceNews: [
                StockNewsItem(
                    title: "Sample Yahoo Finance News with a very long title that should be truncated after three lines of text",
                    date: "2024-01-01",
                    published_at: nil,
                    pubDate: nil,
                    url: "https://example.com"
                ),
                StockNewsItem(
                    title: "Another Yahoo Finance News",
                    date: "2024-01-02",
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
            ]
        ))
        .padding()
        .previewLayout(.sizeThatFits)
    }
}
