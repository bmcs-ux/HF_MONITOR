try:
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5()
    print("Menggunakan mt5linux bridge...")
except ImportError:
    import MetaTrader5 as mt5
    print("Menggunakan MetaTrader5 native...")

mt5.initialize()

symbols = mt5.symbols_get()
print(f"Total symbols: {len(symbols)}")
print([s.name for s in symbols[:20]])

