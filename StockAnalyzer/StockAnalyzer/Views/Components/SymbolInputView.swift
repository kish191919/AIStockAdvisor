
import Foundation
import SwiftUI

// Views/Components/SymbolInputView.swift
struct SymbolInputView: View {
    @Binding var symbol: String
    @ObservedObject var validator: StockSymbolValidator
    let onSubmit: (String) -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Stock Symbol (e.g., AAPL)", text: $symbol)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .autocapitalization(.allCharacters)
                .onChange(of: symbol) { _ in
                    _ = validator.validateSymbolFormat(symbol)
                }
            
            if !validator.isValid {
                Text(validator.errorMessage)
                    .foregroundColor(.red)
                    .font(.caption)
            }
        }
    }
}
