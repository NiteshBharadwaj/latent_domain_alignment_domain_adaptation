def main():
    record_train = '%s/%s_%s.txt' % (args.record_folder, args.target,record_num)
    record_test = '%s/%s_%s_test.txt' % (args.record_folder, args.target, record_num)
    record_val = '%s/%s_%s_val.txt' % (args.record_folder, args.target, record_num)
    checkpoint_dir = '%s/%s_%s' % (args.record_folder, args.target, record_num)
    
    while os.path.exists(record_train):
        record_num += 1
        record_train = '%s/%s_%s.txt' % (args.record_folder, args.target, record_num)
        record_test = '%s/%s_%s_test.txt' % (args.record_folder, args.target, record_num)
        record_val = '%s/%s_%s_val.txt' % (args.record_folder, args.target, record_num)
        checkpoint_dir = '%s/%s_%s' % (args.record_folder, args.target, record_num)
    
    checkpoint_dir = '%s/%s_%s' % (args.record_folder, args.target, record_num-1)
    args.checkpoint_dir = checkpoint_dir 
    
    solver = Solver(args, target=args.target, learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, 
                    checkpoint_dir=args.checkpoint_dir,
                    save_epoch=args.save_epoch)
    test(solver, 0, 'test', record_file=None, save_model=False)
    
    