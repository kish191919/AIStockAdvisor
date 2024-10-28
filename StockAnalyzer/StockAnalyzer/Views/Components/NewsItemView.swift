
//Views/Components/NewsItemView.swift
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

struct NewsItemView: View {
    let item: NewsItem
    
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(item.title)
                .font(.subheadline)
            Text(item.displayDate)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 5)
    }
}
