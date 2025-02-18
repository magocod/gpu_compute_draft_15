; ModuleID = 'global_array-hip-amdgcn-amd-amdhsa-gfx1032.bc'
source_filename = "global_array.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@global_array = protected addrspace(1) externally_initialized global [32 x i32] zeroinitializer, align 16
@__hip_cuid_a15846cea79ef76f = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_a15846cea79ef76f to ptr), ptr addrspacecast (ptr addrspace(1) @global_array to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: read, inaccessiblemem: none)
define protected amdgpu_kernel void @_Z16global_array_putPi(ptr addrspace(1) nocapture noundef readonly %0) local_unnamed_addr #0 {
  %2 = tail call noundef i32 @llvm.amdgcn.workitem.id.x(), !range !6, !noundef !7
  %3 = zext nneg i32 %2 to i64
  %4 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  %5 = load i32, ptr addrspace(1) %4, align 4, !tbaa !8, !amdgpu.noclobber !7
  %6 = getelementptr inbounds [32 x i32], ptr addrspace(1) @global_array, i64 0, i64 %3
  store i32 %5, ptr addrspace(1) %6, align 4, !tbaa !8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none)
define protected amdgpu_kernel void @_Z16global_array_getPi(ptr addrspace(1) nocapture noundef writeonly %0) local_unnamed_addr #1 {
  %2 = tail call noundef i32 @llvm.amdgcn.workitem.id.x(), !range !6, !noundef !7
  %3 = zext nneg i32 %2 to i64
  %4 = getelementptr inbounds [32 x i32], ptr addrspace(1) @global_array, i64 0, i64 %3
  %5 = load i32, ptr addrspace(1) %4, align 4, !tbaa !8
  %6 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %5, ptr addrspace(1) %6, align 4, !tbaa !8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define protected amdgpu_kernel void @_Z21global_array_increasev() local_unnamed_addr #2 {
  %1 = tail call noundef i32 @llvm.amdgcn.workitem.id.x(), !range !6, !noundef !7
  %2 = zext nneg i32 %1 to i64
  %3 = getelementptr inbounds [32 x i32], ptr addrspace(1) @global_array, i64 0, i64 %2
  %4 = load i32, ptr addrspace(1) %3, align 4, !tbaa !8
  %5 = add nsw i32 %4, 1
  store i32 %5, ptr addrspace(1) %3, align 4, !tbaa !8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define protected amdgpu_kernel void @_Z19global_array_insertPiS_(ptr addrspace(1) nocapture noundef readonly %0, ptr addrspace(1) nocapture noundef writeonly %1) local_unnamed_addr #3 {
  %3 = tail call noundef i32 @llvm.amdgcn.workitem.id.x(), !range !6, !noundef !7
  %4 = zext nneg i32 %3 to i64
  %5 = getelementptr inbounds [32 x i32], ptr addrspace(1) @global_array, i64 0, i64 %4
  %6 = load i32, ptr addrspace(1) %5, align 4, !tbaa !8
  %7 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %4
  store i32 %6, ptr addrspace(1) %7, align 4, !tbaa !8
  %8 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %4
  %9 = load i32, ptr addrspace(1) %8, align 4, !tbaa !8
  store i32 %9, ptr addrspace(1) %5, align 4, !tbaa !8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workitem.id.x() #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: read, inaccessiblemem: none) "amdgpu-flat-work-group-size"="1,1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1032" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none) "amdgpu-flat-work-group-size"="1,1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1032" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) "amdgpu-flat-work-group-size"="1,1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1032" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) "amdgpu-flat-work-group-size"="1,1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1032" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!opencl.ocl.version = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"amdgpu_code_object_version", i32 500}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 2, i32 0}
!5 = !{!"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.2.2 24355 77cf9ad00e298ed06e06aec0f81009510f545714)"}
!6 = !{i32 0, i32 1024}
!7 = !{}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
